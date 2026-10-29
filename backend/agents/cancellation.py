"""Cooperative per-request cancellation, including closing active model streams."""
from contextvars import ContextVar
from threading import Event, Lock, Thread
import json
import time
import subprocess
import requests

class GenerationCancelled(BaseException):
    pass

_current = ContextVar('chat_generation', default=None)
_runs = {}
_lock = Lock()

class Generation:
    def __init__(self):
        self.cancelled = Event()
        self.callbacks = []
        self.lock = Lock()
        self.created = time.monotonic()

    def on_cancel(self, callback):
        with self.lock:
            if not self.cancelled.is_set():
                self.callbacks.append(callback)
                return
        Thread(target=_quiet, args=(callback,), daemon=True).start()

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            Thread(target=_quiet, args=(callback,), daemon=True).start()


def _quiet(callback):
    try:
        callback()
    except Exception:
        pass


def register_generation(request_id):
    with _lock:
        for key, run in list(_runs.items()):
            if time.monotonic() - run.created > 7200:
                _runs.pop(key, None)
        return _runs.setdefault(request_id, Generation())


def cancel_generation(request_id):
    # Remember early cancellation even if the stream request has not started yet.
    run = register_generation(request_id)
    run.cancel()


def check_cancelled():
    run = _current.get()
    if run and run.cancelled.is_set():
        raise GenerationCancelled()


def on_cancel(callback):
    run = _current.get()
    if run:
        run.on_cancel(callback)


def cancellable_post(*args, **kwargs):
    check_cancelled()
    response = requests.post(*args, **kwargs)
    run = _current.get()
    if not run:
        return response
    run.on_cancel(response.close)
    check_cancelled()
    original = response.iter_lines
    def lines(*line_args, **line_kwargs):
        try:
            for line in original(*line_args, **line_kwargs):
                check_cancelled()
                yield line
            check_cancelled()
        except Exception:
            check_cancelled()
            raise
    response.iter_lines = lines
    return response


def managed_stream(iterator, request_id, run, initial_session_id=None):
    """Persist stopped partial text without indexing it as verified memory."""
    session_id = initial_session_id
    parts, thinking = [], []
    finished = False
    try:
        while True:
            token = _current.set(run)
            try:
                check_cancelled()
                event = next(iterator)
                check_cancelled()
            except StopIteration:
                return
            finally:
                _current.reset(token)
            try:
                payload = json.loads(event.removeprefix('data:').strip())
                if payload.get('type') == 'session':
                    session_id = payload.get('session_id')
                elif payload.get('type') == 'content':
                    parts.append(payload.get('delta', ''))
                elif payload.get('type') == 'thinking':
                    thinking.append(payload.get('delta', ''))
                elif payload.get('type') == 'done':
                    finished = True
            except (ValueError, TypeError):
                pass
            yield event
    except GenerationCancelled:
        if session_id and not finished:
            from backend.db.models import insert_message
            text = ''.join(parts)
            if thinking:
                text = '<details class="thinking-details"><summary>Partial thinking</summary>\n\n' + ''.join(thinking) + '\n</details>\n\n' + text
            insert_message(session_id=session_id, role='assistant', content=text + '\n\n*Stopped by you. Send your guidance to continue.*',
                           agent_used='Stopped', tool_calls_json=json.dumps({'memory_quality': {'execution_required': True, 'exit_code': None}, 'cancelled': True}))
        yield 'data: ' + json.dumps({'type': 'stopped', 'request_id': request_id, 'session_id': session_id}) + '\n\n'
    finally:
        run.cancel()
        _quiet(getattr(iterator, "close", lambda: None))
        with _lock:
            if _runs.get(request_id) is run:
                _runs.pop(request_id, None)


def run_cancellable_process(command, timeout, cleanup=None):
    """Stop a CLI sandbox process and its named container when cancellation arrives."""
    if _current.get() is None:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    check_cancelled()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.monotonic()
    try:
        while True:
            check_cancelled()
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                check_cancelled()
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        if cleanup:
            _quiet(cleanup)
