import json
import uuid
from threading import Event
import pytest
from backend.agents.cancellation import (register_generation, cancel_generation, managed_stream,
    check_cancelled, GenerationCancelled, cancellable_post, _current)


def test_stop_preserves_partial_and_prevents_next_tool(mocker):
    request_id = str(uuid.uuid4())
    run = register_generation(request_id)
    saved = mocker.patch('backend.db.models.insert_message')
    tools = []
    def agent():
        yield 'data: '+json.dumps({'type': 'session', 'session_id': 42})+'\n\n'
        yield 'data: '+json.dumps({'type': 'content', 'delta': 'partial code'})+'\n\n'
        tools.append('must not execute')
        yield 'data: {"type":"done"}\n\n'
    stream = managed_stream(agent(), request_id, run)
    next(stream); next(stream)
    cancel_generation(request_id)
    assert json.loads(next(stream).removeprefix('data:'))['type'] == 'stopped'
    assert not tools
    assert 'partial code' in saved.call_args.kwargs['content']
    assert json.loads(saved.call_args.kwargs['tool_calls_json'])['cancelled']
    stream.close()


def test_cancel_before_generation_starts(mocker):
    request_id = str(uuid.uuid4())
    cancel_generation(request_id)
    run = register_generation(request_id)
    def agent():
        pytest.fail('Cancelled generation must never start')
        yield
    events = list(managed_stream(agent(), request_id, run))
    assert json.loads(events[0].removeprefix('data:'))['type'] == 'stopped'


def test_cancel_closes_model_connection(mocker):
    request_id = str(uuid.uuid4())
    run = register_generation(request_id)
    closed = Event()
    response = mocker.Mock()
    response.close.side_effect = closed.set
    response.iter_lines.return_value = [b'first', b'second']
    mocker.patch('requests.post', return_value=response)
    token = _current.set(run)
    try:
        stream = cancellable_post('http://model').iter_lines()
        assert next(stream) == b'first'
        cancel_generation(request_id)
        assert closed.wait(2)
        with pytest.raises(GenerationCancelled):
            next(stream)
    finally:
        _current.reset(token)


def test_cancellation_cannot_be_converted_into_repair(mocker):
    from backend.agents.execution_repair import execution_repairs
    request_id = str(uuid.uuid4())
    run = register_generation(request_id)
    repair = mocker.Mock()
    token = _current.set(run)
    try:
        cancel_generation(request_id)
        with pytest.raises(GenerationCancelled):
            list(execution_repairs('print(1)', 'request', mocker.Mock(), repair))
        repair.assert_not_called()
    finally:
        _current.reset(token)


def test_stop_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app
    request_id = str(uuid.uuid4())
    response = TestClient(app).post(f'/api/chat/{request_id}/stop')
    assert response.status_code == 200
    assert register_generation(request_id).cancelled.is_set()
