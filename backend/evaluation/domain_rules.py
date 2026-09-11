from enum import Enum
import re
from typing import Dict, Any, List, Optional

class ScientificDomain(str, Enum):
    SURFACE_AND_INTERFACES = "surface_and_interfaces"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    STRUCTURE_AND_COORDINATION = "structure_and_coordination"
    DFT_AND_ELECTRONIC = "dft_and_electronic"
    THERMODYNAMICS_AND_KINETICS = "thermodynamics_and_kinetics"
    MACHINE_LEARNING_POTENTIALS = "machine_learning_potentials"
    SPECTROSCOPY = "spectroscopy"
    GENERAL_SIMULATION = "general_simulation"

class DomainRule:
    """Represents an algorithmic or physical invariant rule in computational chemistry."""
    def __init__(self, rule_id: str, description: str, weight: int, checker):
        self.rule_id = rule_id
        self.description = description
        self.weight = weight
        self.checker = checker

    def evaluate(self, code: str, ast_info: Dict[str, Any]) -> bool:
        try:
            return bool(self.checker(code, ast_info))
        except Exception:
            return False

class DomainClassifier:
    """
    Classifies scientific computing prompts and code into domain categories
    and supplies specialized physical and programmatic rules for that domain.
    """

    DOMAIN_KEYWORDS: Dict[ScientificDomain, List[str]] = {
        ScientificDomain.SURFACE_AND_INTERFACES: [
            "surface", "slab", "adsorption", "adsorbate", "facet", "coverage", "binding energy",
            "work function", "cus", "overlayer", "vacuum", "interface", "topmost", "surface energy",
            "wulff", "cleave", "terrace", "step edge", "kink", "miller index", "hkl",
            "selective dynamics", "relaxation", "passivation", "reconstruction"
        ],
        ScientificDomain.MOLECULAR_DYNAMICS: [
            "trajectory", "md", "nvt", "npt", "timestep", "msd", "diffusion", "unwrapped",
            "equilibration", "thermostat", "barostat", "velocity", "drift", "extxyz", "lammps",
            "nose-hoover", "langevin", "berendsen", "verlet", "leapfrog", "production run",
            "ensemble", "periodic boundary", "pbc", "shake", "rattle", "time integration"
        ],
        ScientificDomain.STRUCTURE_AND_COORDINATION: [
            "rdf", "radial distribution", "g(r)", "neighbor", "coordination", "voronoi",
            "bond length", "bond angle", "cutoff", "pair distribution", "xrd", "pdf",
            "nearest neighbor", "common neighbor", "steinhardt", "order parameter",
            "topology", "connectivity", "polyhedral", "interatomic distance"
        ],
        ScientificDomain.DFT_AND_ELECTRONIC: [
            "dft", "kpoints", "k-points", "band structure", "dos", "density of states",
            "pseudopotential", "cutoff energy", "encut", "bader", "charge", "vasp", "quantum espresso",
            "gpaw", "siesta", "abinit", "wannier", "eigenvalue", "fermi level", "hartree-fock",
            "exchange-correlation", "pbe", "pbesol", "hybrid functional", "hse", "gw",
            "scf", "self-consistent", "ecutwfc", "ecutrho", "smearing"
        ],
        ScientificDomain.THERMODYNAMICS_AND_KINETICS: [
            "free energy", "survival", "residence time", "tau", "rate constant", "activation energy",
            "arrhenius", "barrier", "reaction coordinate", "umbrella sampling", "kinetics",
            "transition state", "neb", "nudged elastic band", "metadynamics", "thermodynamic integration",
            "free energy perturbation", "entropy", "enthalpy", "gibbs", "boltzmann",
            "partition function", "equilibrium constant"
        ],
        ScientificDomain.MACHINE_LEARNING_POTENTIALS: [
            "deepmd", "deep potential", "nequip", "mace", "allegro", "schnet", "painn",
            "descriptor", "training_data", "validation_loss", "se_e2_a", "fitting_net",
            "machine learning potential", "mlp", "neural network potential", "nnp",
            "moment tensor", "mtp", "gap", "gaussian approximation", "ace",
            "atomic cluster expansion", "force field training", "active learning",
            "dp_train", "dpdata", "model deviation"
        ],
        ScientificDomain.SPECTROSCOPY: [
            "spectrum", "absorption", "ir", "infrared", "raman", "uv-vis", "uv_vis",
            "oscillator strength", "dielectric", "optical", "photoluminescence",
            "fluorescence", "circular dichroism", "nmr", "epr", "xas", "xps",
            "eels", "electron energy loss", "transition dipole", "selection rule",
            "vibrational", "rotational", "electronic transition", "beer-lambert",
            "extinction coefficient", "absorbance", "transmittance"
        ]
    }

    @classmethod
    def classify(cls, prompt: Optional[str], code: str) -> ScientificDomain:
        text = f"{prompt or ''} {code}".lower()
        domain_scores: Dict[ScientificDomain, int] = {d: 0 for d in ScientificDomain}

        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                matches = len(re.findall(r"\b" + re.escape(kw) + r"\b", text))
                domain_scores[domain] += matches

        # Pick the domain with highest score, fallback to GENERAL_SIMULATION
        best_domain, highest_score = max(domain_scores.items(), key=lambda x: x[1])
        if highest_score > 0:
            return best_domain
        return ScientificDomain.GENERAL_SIMULATION

    @classmethod
    def get_rules_for_domain(cls, domain: ScientificDomain) -> List[DomainRule]:
        """Returns standard domain-specific validation rules with granular weights."""
        rules: List[DomainRule] = []

        # Common rule 1: Valid domain library usage
        rules.append(DomainRule(
            "sci_lib_present",
            "Leverages standard computational libraries (ASE, NumPy, SciPy, or Matplotlib)",
            weight=12,
            checker=lambda code, ast_info: len(ast_info.get("patterns", {}).get("scientific_libs", [])) > 0 or
                                          bool(re.search(r"\b(ase|numpy|scipy|matplotlib|pandas|pymatgen|mdanalysis)\b", code, re.IGNORECASE))
        ))

        # Common rule 2: Modular functional decomposition
        rules.append(DomainRule(
            "functional_structure",
            "Defines structured, modular functions or workflow steps",
            weight=10,
            checker=lambda code, ast_info: len(ast_info.get("functions", [])) >= 2 or
                                          code.count("def ") >= 2
        ))

        # Common rule 3: Proper data flow (variables assigned before use)
        rules.append(DomainRule(
            "data_flow_integrity",
            "Maintains proper data flow — variables assigned before use, no dangling references",
            weight=8,
            checker=lambda code, ast_info: len(ast_info.get("undefined_symbols", [])) <= 2
        ))

        # Domain-specific rules
        if domain == ScientificDomain.SURFACE_AND_INTERFACES:
            rules.append(DomainRule(
                "surface_or_slab_logic",
                "Includes surface coordinate filtering, slab generation, or adsorption positioning",
                weight=22,
                checker=lambda code, ast_info: bool(re.search(r"(surface|slab|z_coord|positions\[.*2\]|z\s*>|adsorb|topmost|vacuum)", code, re.IGNORECASE)) and
                                              bool(re.search(r"(\[:.*2\]|positions\[|get_positions|z_coord)", code))
            ))
            rules.append(DomainRule(
                "energy_or_distance_metric",
                "Computes distance, binding energy, or geometric relaxation metric",
                weight=18,
                checker=lambda code, ast_info: bool(re.search(r"(energy|dist|e_ads|binding|cutoff|get_distance|height|get_potential_energy)", code, re.IGNORECASE))
            ))
        elif domain == ScientificDomain.MOLECULAR_DYNAMICS:
            rules.append(DomainRule(
                "trajectory_handling",
                "Handles trajectory frames, indexing, or time evolution sequence",
                weight=22,
                checker=lambda code, ast_info: bool(re.search(r"(traj|read\s*\(|frames?|timestep|step|stride|extxyz|index\s*=\s*['\:])", code, re.IGNORECASE)) and
                                              bool(re.search(r"for\s+\w+\s+in\s+", code))
            ))
            rules.append(DomainRule(
                "dynamical_or_statistical_property",
                "Calculates statistical mechanics observables (MSD, diffusion, temperature, or fluctuations)",
                weight=18,
                checker=lambda code, ast_info: bool(re.search(r"(msd|diffusion|mean|std|fluctuation|drift|velocity|temperature|p_t|autocorrelation)", code, re.IGNORECASE))
            ))
        elif domain == ScientificDomain.STRUCTURE_AND_COORDINATION:
            rules.append(DomainRule(
                "distance_or_cutoff_computation",
                "Implements neighbor search, pair distances, or cutoff radii",
                weight=20,
                checker=lambda code, ast_info: bool(re.search(r"(neighbor|cutoff|distance|dr|r_max|bins|rpdf|get_all_distances)", code, re.IGNORECASE))
            ))
            rules.append(DomainRule(
                "distribution_or_coordination_metric",
                "Calculates distribution function g(r), histogram, or coordination count with normalization",
                weight=20,
                checker=lambda code, ast_info: bool(re.search(r"(g_r|rdf|hist|coordination|bin|distribution|density)", code, re.IGNORECASE)) and
                                              bool(re.search(r"(/\s*\(?\s*4\s*\*?\s*(?:np\.)?pi|normalize|4.*pi.*r\*\*2|shell_volume)", code, re.IGNORECASE))
            ))
        elif domain == ScientificDomain.DFT_AND_ELECTRONIC:
            rules.append(DomainRule(
                "dft_parameter_specification",
                "Configures electronic structure parameters (k-points, calculator, or basis cutoff)",
                weight=22,
                checker=lambda code, ast_info: bool(re.search(r"(calculator|kpts|kpoints|encut|cutoff|vasp|espresso|gpaw|xc|pbe|ecutwfc)", code, re.IGNORECASE))
            ))
            rules.append(DomainRule(
                "convergence_or_property_extraction",
                "Extracts total energy, forces, band gap, or eigenvalues",
                weight=18,
                checker=lambda code, ast_info: bool(re.search(r"(get_potential_energy|energy|forces|bandgap|dos|fermi|bader|eigenvalue)", code, re.IGNORECASE))
            ))
        elif domain == ScientificDomain.THERMODYNAMICS_AND_KINETICS:
            rules.append(DomainRule(
                "kinetic_or_residence_formalism",
                "Formulates decay curves, survival counts, rates, or time integration",
                weight=22,
                checker=lambda code, ast_info: bool(re.search(r"(tau|residence|survival|rate|decay|arrhenius|barrier|delta_g|trapz|trapezoid)", code, re.IGNORECASE))
            ))
            rules.append(DomainRule(
                "thermodynamic_parameter_sanity",
                "Respects thermodynamic constraints (positive temperatures, finite rates, or normalized fractions)",
                weight=18,
                checker=lambda code, ast_info: bool(re.search(r"(fraction|pct|exp|log|k_b|temperature|boltzmann|integration|kT)", code, re.IGNORECASE))
            ))
        elif domain == ScientificDomain.MACHINE_LEARNING_POTENTIALS:
            rules.append(DomainRule(
                "mlp_architecture_config",
                "Configures ML potential architecture (descriptor, fitting network, or model hyperparameters)",
                weight=22,
                checker=lambda code, ast_info: bool(re.search(r"(descriptor|se_e2_a|fitting_net|neuron|rcut|sel\s*[=:\[]|embedding|model\s*[=:{])", code, re.IGNORECASE))
            ))
            rules.append(DomainRule(
                "training_data_or_loss",
                "Specifies training data paths, loss function, or validation configuration",
                weight=18,
                checker=lambda code, ast_info: bool(re.search(r"(training|validation|loss|pref_e|pref_f|batch_size|learning_rate|numb_steps|epoch)", code, re.IGNORECASE))
            ))
        elif domain == ScientificDomain.SPECTROSCOPY:
            rules.append(DomainRule(
                "spectral_computation",
                "Computes spectral quantities (frequencies, intensities, cross-sections, or dielectric response)",
                weight=22,
                checker=lambda code, ast_info: bool(re.search(r"(frequency|wavenumber|intensity|cross.section|oscillator|dipole|dielectric|epsilon|polarizability)", code, re.IGNORECASE))
            ))
            rules.append(DomainRule(
                "spectral_presentation",
                "Generates spectral plots with proper axes (energy/frequency vs intensity/absorption)",
                weight=18,
                checker=lambda code, ast_info: bool(re.search(r"(spectrum|peak|broadening|gaussian|lorentzian|convolv|fwhm|cm\^?-1|eV|THz)", code, re.IGNORECASE))
            ))
        else: # GENERAL_SIMULATION
            rules.append(DomainRule(
                "data_manipulation_or_modeling",
                "Applies numerical transformations, array slicing, or simulation routines",
                weight=20,
                checker=lambda code, ast_info: bool(re.search(r"(np\.|array|zeros|arange|linspace|filter|transform|calc)", code, re.IGNORECASE))
            ))
            rules.append(DomainRule(
                "quantitative_metric_extraction",
                "Computes quantitative physical or computational values",
                weight=20,
                checker=lambda code, ast_info: bool(re.search(r"(mean|sum|std|min|max|diff|result|value)", code, re.IGNORECASE))
            ))

        # Common rule 4: Output persistence or visualization
        rules.append(DomainRule(
            "visualization_or_output_artifacts",
            "Generates visual figures (matplotlib) or output data files (.png, .txt, .csv, .npz)",
            weight=10,
            checker=lambda code, ast_info: ast_info.get("patterns", {}).get("has_plotting", False) or
                                          ast_info.get("patterns", {}).get("has_saving", False) or
                                          bool(re.search(r"(plt\.|savefig|to_csv|savez|print\s*\(|with open)", code, re.IGNORECASE))
        ))

        return rules
