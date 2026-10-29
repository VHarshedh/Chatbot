"""Shared bounded execution/repair loop for every code-executing agent."""
import re
import json
from backend.agents.cancellation import check_cancelled
from backend.tools.validation_tools import local_error_context, execution_error_messages

MAX_DEBUG_ITERATIONS = 5


def stream_repair_response(response):
    """Forward only model-provided thinking and code; support JSON-only endpoints."""
    received = False
    try:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            received = True
            msg = chunk.get("message", {})
            for field in ("thinking", "content"):
                if msg.get(field):
                    yield {"type": field, "delta": msg[field]}
        if not received:
            data = response.json()
            if data.get("error"):
                raise RuntimeError(data["error"])
            msg = data.get("message", {})
            for field in ("thinking", "content"):
                value = msg.get(field) or (data.get("response", "") if field == "content" else "")
                if value:
                    yield {"type": field, "delta": value}
    finally:
        response.close()


def execution_repairs(code, user_prompt, execute, repair, max_repairs=MAX_DEBUG_ITERATIONS):
    """Initial execution plus at most five model repairs, each executed and audited.

    Yields progress events and a final result containing the last executed code.
    execute must use run_python_in_sandbox, including its scientific output audit.
    """
    failures = []
    for attempt in range(max_repairs + 1):
        check_cancelled()
        yield {'type': 'agent_step', 'step': 'sandbox_start', 'title': 'Docker execution', 'status': 'running',
               'attempt': attempt + 1, 'max_attempts': max_repairs + 1, 'details': f'Execution {attempt + 1}/{max_repairs + 1}'}
        try:
            result = execute(code)
        except Exception as exc:
            result = {'exit_code': 1, 'stdout': '', 'stderr': str(exc), 'artifacts': []}
        # Inspect raw streams even if a runner mistakenly marks a caught exception successful.
        reported_errors = execution_error_messages(result.get('stdout', ''), result.get('stderr', ''))
        if reported_errors and result.get('exit_code') == 0:
            result = {**result, 'process_exit_code': 0, 'exit_code': 3,
                      'stderr': result.get('stderr', '') + '\nExecution reported errors: ' + '; '.join(reported_errors)}
        if result.get('audit', {}).get('valid') is False and result.get('exit_code') == 0:
            result = {**result, 'process_exit_code': 0, 'exit_code': 3,
                      'stderr': result.get('stderr', '') + '\n' + '; '.join(result['audit'].get('issues', []))}
        valid = result.get('exit_code') == 0
        yield {'type': 'agent_step', 'step': 'sandbox_complete', 'title': 'Docker execution',
               'status': 'completed' if valid else 'warning', 'attempt': attempt + 1,
               'details': 'Execution and output checks passed.' if valid else str(result.get('stderr', 'Execution failed'))[:500]}
        if valid or attempt == max_repairs:
            yield {'type': 'execution_result', 'result': result, 'code': code, 'attempts': attempt + 1,
                   'repairs': attempt, 'exhausted': not valid}
            return
        failures.append({'attempt': attempt + 1, 'exit_code': result.get('exit_code'),
                         'stdout': result.get('stdout', '')[-3000:], 'stderr': result.get('stderr', '')[-4000:],
                         'audit': result.get('audit', {})})
        yield {'type': 'agent_step', 'step': 'sandbox_retry', 'title': 'Debug and repair', 'status': 'running',
               'iteration': attempt + 1, 'details': f'Debugging iteration {attempt + 1}/{max_repairs}'}
        yield {'type': 'agent_step', 'step': 'local_api', 'title': 'Local Python API lookup',
               'status': 'running', 'iteration': attempt + 1, 'tool': 'inspect_local_python_api',
               'details': 'Inspecting installed APIs for the failed script.'}
        api_context = local_error_context(code, result.get('stderr', ''))
        yield {'type': 'agent_step', 'step': 'local_api', 'title': 'Local Python API lookup',
               'status': 'completed', 'iteration': attempt + 1, 'tool': 'inspect_local_python_api',
               'details': api_context[:2000] or 'No applicable installed API reference found.'}
        prompt = f'''Correct the complete Python script to fulfill the original request.
Original request:
{user_prompt}

Execution/validation failures (do not repeat unsuccessful fixes):
{failures}

Installed API reference:
{api_context}

Last executed script:
```python
{code}
```
Return only a complete corrected script in a python code fence. Fix runtime errors AND output audit failures. Do not suppress errors, remove requested outputs, or fabricate results.'''
        try:
            check_cancelled()
            response = repair(prompt)
            raw = response if isinstance(response, str) else ""
            streamed_content = False
            if not isinstance(response, str):
                yield {'type': 'content', 'iteration': attempt + 1, 'phase': 'repair',
                       'delta': f"\n\n### Self-Corrected Script (repair {attempt + 1}/{max_repairs})\n"}
                for event in response:
                    event = {**event, 'iteration': attempt + 1, 'phase': 'repair'}
                    if event['type'] == 'content':
                        raw += event.get('delta', '')
                        streamed_content = True
                    yield event
            match = re.search(r'```(?:python)?\s*([\s\S]*?)```', raw)
            candidate = match.group(1).strip() if match else raw.strip()
            if not candidate:
                raise ValueError('The model returned no corrected script')
        except Exception as exc:
            yield {'type': 'execution_result', 'result': {**result, 'stderr': result.get('stderr', '') + '\nRepair failed: ' + str(exc)},
                   'code': code, 'attempts': attempt + 1, 'repairs': attempt, 'exhausted': True}
            return
        code = candidate
        yield {'type': 'repaired_code', 'code': code, 'iteration': attempt + 1, 'streamed': streamed_content}
