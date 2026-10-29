from backend.agents.execution_repair import execution_repairs


def test_repairs_until_sixth_execution_succeeds():
    calls, prompts = [], []
    def execute(code):
        calls.append(code)
        return {'exit_code': 0 if len(calls) == 6 else 3, 'stdout': 'wrong data', 'stderr': 'output invalid', 'artifacts': []}
    def repair(prompt):
        prompts.append(prompt)
        return f'```python\nprint({len(prompts)})\n```'
    events = list(execution_repairs('print(0)', 'calculate density', execute, repair))
    assert len(calls) == 6 and len(prompts) == 5
    assert events[-1]['result']['exit_code'] == 0
    assert events[-1]['code'] == 'print(5)'
    assert 'wrong data' in prompts[-1] and 'calculate density' in prompts[-1]


def test_stops_after_five_repairs_and_keeps_last_executed_code():
    events = list(execution_repairs('print(0)', 'request',
        lambda code: {'exit_code': 1, 'stderr': 'bad', 'artifacts': []}, lambda prompt: 'print(1)'))
    assert events[-1]['attempts'] == 6
    assert events[-1]['repairs'] == 5
    assert events[-1]['exhausted']
    assert len([e for e in events if e['type'] == 'repaired_code']) == 5


def test_semantic_failure_is_repaired_despite_zero_process_exit():
    results = iter([{'exit_code': 0, 'audit': {'valid': False, 'issues': ['missing plot']}, 'stderr': ''}, {'exit_code': 0}])
    events = list(execution_repairs('print(0)', 'plot', lambda code: next(results), lambda prompt: 'print(1)'))
    assert events[-1]['attempts'] == 2
    assert events[-1]['result']['exit_code'] == 0


def test_success_does_not_call_repair():
    def repair(prompt):
        raise AssertionError('unnecessary repair')
    events = list(execution_repairs('print(0)', 'request', lambda code: {'exit_code': 0}, repair))
    assert events[-1]['attempts'] == 1


def test_empty_repair_returns_last_executed_failure():
    events = list(execution_repairs('bad()', 'request', lambda code: {'exit_code': 1, 'stderr': 'bad'}, lambda prompt: ''))
    assert events[-1]['code'] == 'bad()'
    assert events[-1]['result']['exit_code'] != 0


def test_caught_forces_error_triggers_repair_even_with_success_audit():
    error = 'Successfully loaded structure\nAn error occurred during processing: The property "forces" is not available.'
    results = iter([{'exit_code': 0, 'stdout': error, 'stderr': '', 'audit': {'valid': True}},
                    {'exit_code': 0, 'stdout': 'Boundary atoms: 24', 'stderr': ''}])
    prompts = []
    def repair(prompt):
        prompts.append(prompt)
        return 'print("Boundary atoms: 24")'
    events = list(execution_repairs('atoms.get_forces()', 'Find boundary coordinates', lambda code: next(results), repair))
    assert events[-1]['attempts'] == 2
    assert events[-1]['result']['exit_code'] == 0
    assert 'The property "forces" is not available' in prompts[0]
    assert 'never fabricate forces' in prompts[0]
    warning = next(e for e in events if e.get('step') == 'sandbox_complete')
    assert warning['status'] == 'warning'


def test_error_reporting_in_either_stream_fails_audit():
    from backend.tools.validation_tools import audit_sandbox_execution_result
    messages = ['An error occurred during processing: forces unavailable',
                'ValueError: bad shape', 'Error while loading data',
                'Traceback (most recent call last):', 'Processing failed: missing data']
    for message in messages:
        for stdout, stderr in [(message, ''), ('', message)]:
            assert not audit_sandbox_execution_result(stdout, stderr, [])['valid']
    for message in ['Mean absolute error: 0.01', 'Error estimate = 0.2', 'No errors occurred.', 'DeprecationWarning: old API']:
        assert audit_sandbox_execution_result(message, '', [])['valid']


def test_caught_error_wrapper_marks_process_zero_as_failed(mocker):
    from backend.tools.sandbox_tools import run_python_in_sandbox
    result = {'exit_code': 0, 'stdout': 'An error occurred during processing: The property "forces" is not available.', 'stderr': '', 'artifacts': []}
    mocker.patch('backend.tools.sandbox_tools.SandboxExecutor').return_value.run_python_script.return_value = result
    audited = run_python_in_sandbox('print("caught exception")')
    assert audited['exit_code'] == 3
    assert audited['process_exit_code'] == 0
    assert not audited['audit']['valid']


def test_streamed_repairs_forward_thinking_code_and_tools_before_reexecution():
    timeline = []
    def execute(code):
        timeline.append(('execute', code))
        return {'exit_code': 1, 'stderr': 'ValueError: bad data'} if len(timeline) == 1 else {'exit_code': 0}
    def repair(prompt):
        yield {'type': 'thinking', 'delta': 'Checking the reported input shape.'}
        yield {'type': 'agent_step', 'step': 'web_lookup', 'title': 'Web lookup', 'status': 'completed', 'details': 'actual tool result'}
        yield {'type': 'content', 'delta': '```python\n'}
        yield {'type': 'content', 'delta': 'print("fixed")\n```'}
    events = list(execution_repairs('bad()', 'fix input', execute, repair))
    thinking = next(e for e in events if e['type'] == 'thinking')
    assert thinking['iteration'] == 1 and thinking['phase'] == 'repair'
    assert any(e.get('step') == 'local_api' and e['status'] == 'completed' for e in events)
    assert any(e.get('step') == 'web_lookup' for e in events)
    final = events[-1]
    assert final['code'] == 'print("fixed")'
    assert final['attempts'] == 2
    code_event = next(e for e in events if e['type'] == 'repaired_code')
    assert code_event['streamed'] is True


def test_repair_response_stream_and_json_fallback(mocker):
    import json
    from backend.agents.execution_repair import stream_repair_response
    response = mocker.Mock()
    response.iter_lines.return_value = [
        json.dumps({'message': {'thinking': 'diagnosis'}}).encode(),
        json.dumps({'message': {'content': 'print(1)'}}).encode(),
    ]
    assert list(stream_repair_response(response)) == [
        {'type': 'thinking', 'delta': 'diagnosis'}, {'type': 'content', 'delta': 'print(1)'}]
    response.close.assert_called_once()
    response.iter_lines.return_value = []
    response.json.return_value = {'message': {'thinking': 'JSON diagnosis', 'content': 'print(2)'}}
    assert len(list(stream_repair_response(response))) == 2


def test_repair_response_closes_on_error(mocker):
    import pytest
    from backend.agents.execution_repair import stream_repair_response
    response = mocker.Mock()
    response.iter_lines.return_value = [b'{"error": "model unavailable"}']
    with pytest.raises(RuntimeError, match='model unavailable'):
        list(stream_repair_response(response))
    response.close.assert_called_once()
