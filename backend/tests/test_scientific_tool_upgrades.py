import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from backend.tools.analysis_tools import *
from backend.tools.training_tools import validate_raw_data, create_test_split, raw_to_set
from backend.tools.validation_tools import *


def water_slab():
    return Atoms('TiO2OH2', positions=[[0,0,1],[1,0,1],[0,1,1],[5,5,5],[5.95,5,5],[4.7,5.9,5]], cell=[10,10,10], pbc=[True,True,False])


def test_partition_and_inspection(tmp_path):
    atoms = water_slab()
    assert separate_water_and_crystal(atoms)['crystal_indices'] == [0,1,2]
    assert find_enclosing_crystal_boundary(atoms) == [0,1,2]
    path = tmp_path/'slab.extxyz'
    write(path, [atoms, atoms])
    meta = inspect_structure_metadata(str(path))
    assert meta['frame_count'] == 2
    assert meta['pbc'] == [True,True,False]
    assert meta['phase_partition_preview']['water_oxygen_count'] == 1
    assert identify_sites(atoms, 'Ti', 2)['stoichiometry_matches']


def test_density_conserves_mass():
    atoms = water_slab()
    result = compute_density_profile([atoms], 1, atoms.cell)
    assert sum(result['density'])*100/1.66053906660 == pytest.approx(sum(atoms.get_masses()))


def test_real_rdf_and_survival():
    atoms = Atoms('HH', positions=[[0,0,0],[1,0,0]], cell=[10,10,10], pbc=True)
    rdf = compute_pair_rdf([atoms], [('H','H')], .2)
    assert np.count_nonzero(rdf['g_r']) == 1
    assert rdf['r'][np.argmax(rdf['g_r'])] == pytest.approx(1.1)
    frames = [water_slab(), water_slab()]
    frames[1].positions[4:] += [0,0,3]
    assert compute_survival_probability(frames, 'O_w', 2)['p_t'].tolist() == [1,0]
    assert compute_hydroxyl_fraction([frames[0]], 1.25)['fraction'].tolist() == [0]


def make_raw(path):
    np.savetxt(path/'type.raw', [0,1], fmt='%d')
    for name, width in [('box',9), ('coord',6), ('force',6), ('energy',1)]:
        np.savetxt(path/(name+'.raw'), np.arange(5*width).reshape(5,width))


def test_training_split_and_conversion(tmp_path):
    make_raw(tmp_path)
    assert validate_raw_data(tmp_path)['valid']
    create_test_split(tmp_path, 2)
    train = np.loadtxt(tmp_path/'Train/frame_indices.txt', ndmin=1)
    test = np.loadtxt(tmp_path/'Test/frame_indices.txt', ndmin=1)
    assert set(train).isdisjoint(test)
    assert sorted([*train,*test]) == list(range(5))
    paths = raw_to_set(tmp_path/'Train', 2)
    assert len(paths) == 2
    assert np.load(Path(paths[1])/'coord.npy').shape == (1,6)
    np.savetxt(tmp_path/'force.raw', [[1]])
    assert not validate_raw_data(tmp_path)['valid']


def test_static_checks_without_false_tuple_warning():
    assert validate_scientific_script_ast("x = neighbor_list('ij', atoms, 1); y = x[0]")['valid']
    assert not validate_scientific_script_ast("x = neighbor_list('ij', atoms, 1); y = x[:,0]")['valid']
    assert not validate_scientific_script_ast("atoms.get_neighbors()")['valid']
    assert not validate_scientific_script_ast("plt.scatter(p[:,0],p[:,1])", 'side view')['valid']
    assert validate_scientific_script_ast("nl.get_neighbors(0)")['valid']
    assert 'neighbor_list' in inspect_local_python_api('ase.neighborlist.neighbor_list')


def test_output_gate(tmp_path):
    assert not audit_sandbox_execution_result('No boundary atoms found', '', [], '')['valid']
    assert not audit_sandbox_execution_result('', '', [], 'save boundary.png')['valid']
    assert not audit_sandbox_execution_result('Boundary atoms: 9', '', [], '', {'phase_partition_preview': {'crystal_slab_atom_count': 3}})['valid']
    assert audit_sandbox_execution_result('Boundary atoms: 3', '', [], '', {'phase_partition_preview': {'crystal_slab_atom_count': 3}})['valid']


def test_static_gate_precedes_docker(monkeypatch):
    from backend.tools import sandbox_tools
    monkeypatch.setattr(sandbox_tools, 'SandboxExecutor', lambda: pytest.fail('Docker must not start'))
    assert sandbox_tools.run_python_in_sandbox('def invalid(')['exit_code'] == 2


def test_sandbox_separates_streams_and_mounts_inputs(mocker, tmp_path):
    from backend.sandbox.executor import SandboxExecutor
    executor = SandboxExecutor.__new__(SandboxExecutor)
    executor.image_name = 'test-image'
    executor.wsl_available = False
    executor.client = mocker.MagicMock()
    container = executor.client.containers.run.return_value
    container.wait.return_value = {'StatusCode': 0}
    container.logs.side_effect = lambda **kw: b'actual output' if kw['stdout'] else b'DeprecationWarning: example'
    inputs = [tmp_path/'a.xyz', tmp_path/'b.xyz']
    for path in inputs:
        path.write_text('input')
    result = executor.run_python_script('print(1)', list(map(str, inputs)))
    assert result['stdout'] == 'actual output'
    assert result['stderr'] == 'DeprecationWarning: example'
    mounts = executor.client.containers.run.call_args.kwargs['volumes']
    assert all(mounts[str(path)]['mode'] == 'ro' for path in inputs)


def test_output_failure_overrides_success(mocker):
    from backend.tools.sandbox_tools import run_python_in_sandbox
    executor = mocker.patch('backend.tools.sandbox_tools.SandboxExecutor').return_value
    executor.run_python_script.return_value = {'exit_code': 0, 'stdout': 'No boundary atoms', 'stderr': '', 'artifacts': []}
    result = run_python_in_sandbox("print('No boundary atoms')")
    assert result['exit_code'] == 3
    assert result['process_exit_code'] == 0


def test_rag_budget_and_siblings(mocker):
    from backend.rag.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.kg = mocker.MagicMock()
    retriever.kg.linked_facts.return_value = ['TiO2 reference']
    retriever.kg.query_physchem_rules.return_value = []
    retriever.kg.structure_metadata.return_value = {}
    retriever.vector_store = mocker.MagicMock()
    retriever.vector_store.semantic_search.return_value = [{'id': 'cell', 'content': 'positions = atoms.positions', 'metadata': {'parent_id': 'parent', 'cell_index': 2}}]
    retriever.vector_store.notebook_siblings.return_value = 'import numpy as np'
    retriever.chat_memory = mocker.MagicMock()
    retriever.chat_memory.search_memory.return_value = []
    context = retriever.retrieve('TiO2 positions')
    assert 'import numpy' in context
    assert len(context.encode('utf-8')) <= 1800
    assert context.endswith('</retrieved_context>')


def test_no_fake_slurm_job_id(mocker):
    from backend.tools.slurm_tools import submit_job
    mocker.patch('backend.tools.slurm_tools._run_ssh_cmd', return_value='')
    with pytest.raises(RuntimeError, match='job ID'):
        submit_job('job.sh')


def test_native_cli_fallback_keeps_isolation(mocker):
    import subprocess
    from backend.sandbox.executor import SandboxExecutor
    mocker.patch('docker.from_env', side_effect=RuntimeError('SDK pipe unavailable'))
    mocker.patch('shutil.which', side_effect=lambda name: 'docker.exe' if name == 'docker' else None)
    run = mocker.patch('subprocess.run', side_effect=[
        subprocess.CompletedProcess([], 0, '29.8.2', ''),
        subprocess.CompletedProcess([], 0, 'cli result', 'cli diagnostic'),
    ])
    executor = SandboxExecutor()
    result = executor.run_python_script('print(1)')
    assert result['stdout'] == 'cli result'
    assert result['stderr'] == 'cli diagnostic'
    args = run.call_args.args[0]
    assert args[:2] == ['docker', 'run']
    assert '--read-only' in args
    assert args[args.index('--network')+1] == 'none'
