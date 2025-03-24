"""
This code was copied from here https://github.com/deepchem/deepchem/blob/73c877a79d784220eec375aebae89d6efa71bfc8/deepchem/splits/splitters.py#L1481-L1615
"""

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from .utils import remove_non_mols
import pandas as pd
from rdkit import Chem
import torch_geometric


# class ScaffoldSplitter(Splitter):
class ScaffoldSplitter:
    """Class for doing data splits based on the scaffold of small molecules.

    Group  molecules  based on  the Bemis-Murcko scaffold representation, which identifies rings,
    linkers, frameworks (combinations between linkers and rings) and atomic properties  such as
    atom type, hibridization and bond order in a dataset of molecules. Then split the groups by
    the number of molecules in each group in decreasing order.

    It is necessary to add the smiles representation in the ids field during the
    DiskDataset creation.

    Examples
    ---------
    >>> import deepchem as dc
    >>> # creation of demo data set with some smiles strings
    ... data_test= ["CC(C)Cl" , "CCC(C)CO" ,  "CCCCCCCO" , "CCCCCCCC(=O)OC" , "c3ccc2nc1ccccc1cc2c3" , "Nc2cccc3nc1ccccc1cc23" , "C1CCCCCC1" ]
    >>> Xs = np.zeros(len(data_test))
    >>> Ys = np.ones(len(data_test))
    >>> # creation of a deepchem dataset with the smile codes in the ids field
    ... dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(data_test)),ids=data_test)
    >>> scaffoldsplitter = dc.splits.ScaffoldSplitter()
    >>> train,test = scaffoldsplitter.train_test_split(dataset)
    >>> train
    <DiskDataset X.shape: (5,), y.shape: (5,), w.shape: (5,), ids: ['CC(C)Cl' 'CCC(C)CO' 'CCCCCCCO' 'CCCCCCCC(=O)OC' 'C1CCCCCC1'], task_names: [0]>

    References
    ----------
    .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
        1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

    Notes
    -----
    - This class requires RDKit to be installed.

    - When a SMILES representation of a molecule is invalid, the splitter skips processing
    the datapoint i.e it will not include the molecule in any splits.

    """

    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = 1000,
        include_chirality=True,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float, optional (default 0.8)
            The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
            The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
            The fraction of data to be used for the test split.
        seed: int, optional (default None)
            Random seed to use.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
            Each indices is a list of integers.
        """

        if isinstance(dataset, pd.DataFrame):
            dataset.reset_index(drop=True, inplace=True)
            smiles = dataset.smiles.values
        elif isinstance(dataset, torch_geometric.datasets.molecule_net.MoleculeNet):
            smiles = [i.smiles for i in dataset]

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        scaffold_sets = self.generate_scaffolds(
            smiles, include_chirality=include_chirality
        )

        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        print("About to sort in scaffold sets")
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set

        if isinstance(dataset, pd.DataFrame):
            df_train = dataset.loc[train_inds, :]
            df_val = dataset.loc[valid_inds, :]
            df_test = dataset.loc[test_inds, :]

            df_train.reset_index(drop=True, inplace=True)
            df_val.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)

        if isinstance(dataset, torch_geometric.datasets.molecule_net.MoleculeNet):
            df_train = dataset.index_select(train_inds)
            df_val = dataset.index_select(valid_inds)
            df_test = dataset.index_select(test_inds)

        # , train_inds, valid_inds, test_inds
        return df_train, df_val, df_test

    def generate_scaffolds(
        self, smiles_list, include_chirality, log_every_n: int = 1000
    ) -> List[List[int]]:
        """Returns all scaffolds from the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        scaffold_sets: List[List[int]]
            List of indices of each scaffold in the dataset.
        """
        scaffolds = {}
        data_len = len(smiles_list)

        print("About to generate scaffolds")
        for ind, smiles in enumerate(smiles_list):
            if ind % log_every_n == 0:
                print("Generating scaffold %d/%d" % (ind, data_len))
            scaffold = _generate_scaffold(smiles, include_chirality)
            if scaffold is not None:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [ind]
                else:
                    scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]
        return scaffold_sets


def _generate_scaffold(
    smiles: str, include_chirality: bool = False
) -> Union[str, None]:
    """Compute the Bemis-Murcko scaffold for a SMILES string.

    Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
    They are essentially that part of the molecule consisting of
    rings and the linker atoms between them.

    Paramters
    ---------
    smiles: str
        SMILES
    include_chirality: bool, default False
        Whether to include chirality in scaffolds or not.

    Returns
    -------
    str
        The MurckScaffold SMILES from the original SMILES

    References
    ----------
    .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
        1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

    Note
    ----
    This function requires RDKit to be installed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.info(
            "Not generating scaffold for smiles %s - invalid smiles string" % smiles
        )
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold
