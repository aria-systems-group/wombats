"""
This script contains estimators for solving automata inference problem.
The classes should only depend on data.
Additionally, you can choose to provide specification & parameters used
to generate the data.
"""

# Author: Kandai Watanabe <kandai.wata@gmail.com>

import os
import queue
from tkinter import W
import warnings
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
from wombats.automaton import Automaton, DFA, SafetyDFA, PDFA, Product
from wombats.automaton import active_automata
from .external_tools import FlexfringeInterface
from wombats.automaton.types import Symbols, Probabilities
from wombats.learning.dataload import Dataset

import gurobipy as gpy
from gurobipy import GRB

State = List[float]
Mode = int
HybridState = Tuple[State, Mode]
HybridStateTraj = List[HybridState]
HybridStateTrajs = List[HybridStateTraj]
StateTraj = List[State]
StateTrajs = List[StateTraj]
ModeTraj = List[Mode]
ModeTrajs = List[ModeTraj]


class DummyEstimator(BaseEstimator):
    """
    This class is a dummy estimator for defining a sklearn pipeline.
    The actual estimators will be provided to a GridSearch
    as a list of parameters and replaced with the dummy
    estimator during the fitting process.
    """
    def fit(self, X, y=None, **kwargs): pass
    def score(self, X) -> List[float]: return len(X)*[0.0]


class SpecificationEstimator(BaseEstimator, metaclass=ABCMeta):
    """
    This is the base class for specification (Automata) learning
    estimator.
    """

    key = 'clf'

    def __init__(self, specification: Union[PDFA, Product] = None,
                 output_directory: str = './',
                 binary_location: str = 'dfasat/flexfringe'):
        """
        :param specification:           A PDFA or Product Automaton.
                                        Specification does not have to
                                        be provided, but if it did, then
                                        it makes the computation
                                        slightly faster.
        :param output_directory:        An output directory for
                                        exporting data for flexfringe
        """
        self.specification = specification
        self.output_directory = output_directory
        self.binary_location = binary_location

        self.default_params = {
            'h': 'kldistance',
            'd': 'kl_data',
            'n': '2',
            'x': '0',
            'w': '0',
            'b': '1',
            'f': '1',
            'I': '0',
            't': '1',
            'l': '0',
            'q': '0',
            'y': '0',
            'T': '1',
            'p': '1'}

        for k, v in self.default_params.items():
             setattr(self, k, v)

        self.flexfringe = None
        """FlexfringeInterface"""

        self.params = {self.key: self.__class__.__name__}
        """Parameters that were modified by the user"""

        self.pdfa = None
        """An estimate PDFA from provided data"""

        self.dataset = None
        """Dataset used to train this estimator"""

    def get_param_kwargs(self) -> Dict:
        """
        Update kwargs for passing to FlexfringeInterface
        Assumption: options' lengths must be 1 alphabet
        """
        kwargs = {}
        for key in self.default_params.keys():
            val = getattr(self, key)
            kwargs.update({key: val})
        return kwargs

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :param deep:            If True, will return the parameters for
                                this estimator and contained subobjects that are estimators.
        :return:                Parameter names mapped to their values.
        """
        out = dict()
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value

        return out

    def set_params(self, **params) -> None:
        """
        Set parameters for this estimator

        :param **params:        A dict of params

        :return:                None
        """
        for key, val in params.items():
            if key in self.default_params.keys():
                self.params.update({key: val})
        super().set_params(**params)

    @abstractmethod
    def fit(self, X: List[Symbols], y=None, **kwargs) -> None:
        """
        :param X:           A list of traces datasets
        :param y:           Never be provided. Just being consistent
                            with the base class

        :return PDFA:       A learned PDFA
        """
        raise NotImplementedError('')

    def predict(self, X: List[Symbols]) -> Probabilities:
        """
        Run prediction on the dataset X to compute the scores y

        :param X:       A list of traces

        :return y:      A list of probabilities
        """
        if self.predict is None:
            raise ReferenceError('Not fitted yet')
        return self.pdfa.scores(X)

    def _preprocess(self, X: List[Symbols], filename: str = 'train.abbadingo') -> str:
        """
        Preprocess step before running inference

        :param X:       A list of traces

        :return:        A filename of the exported training data
        """
        if self.specification is None:
            symbols = [symbol for trace in X for symbol in trace]
            alphabet_size = len(set(symbols))
        else:
            alphabet_size = self.specification.alphabet_size

        self.flexfringe = FlexfringeInterface(
            output_directory=self.output_directory,
            binary_location=self.binary_location)

        train_data_file = Automaton.write_traces_to_file(X,
            alphabet_size=alphabet_size,
            file=os.path.join('flexfringe_data', filename))

        return train_data_file

    def _postprocess(self, X: List[Symbols],
                     pdfa: Union[PDFA, Product]) -> None:
        """
        Postprocess step after running inference

        :param X:           A list of traces
        :param pdfa:        A learned PDFA
        :param flexfringe:  An interface to flexfringe library

        :return:            None
        """
        self.pdfa = pdfa

        if self.dataset is None:
            self.dataset = Dataset(X, specification=self.specification)


class Vanilla(SpecificationEstimator):
    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """
        train_data_file = self._preprocess(X)

        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = self.flexfringe.infer_model(
                    training_file=train_data_file,
                    record_time=True, **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except:
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class Postprocess(SpecificationEstimator):
    def __init__(self, safe_specification: SafetyDFA = None, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)
        self.safe_specification = safe_specification

    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """

        train_data_file = self._preprocess(X)

        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = self.flexfringe.infer_model(
                    training_file=train_data_file,
                    record_time=True, **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                # Postprocessing Safety
                if self.safe_specification:
                    pdfa = active_automata.get(
                        automaton_type='PDFA',
                        graph_data=(pdfa, self.safe_specification),
                        graph_data_format='existing_objects',
                        normalize_trans_probabilities=True,
                        delete_sinks=False)

                success = True
            except:
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class GreedyPreprocess(SpecificationEstimator):
    def __init__(self, safe_specification: SafetyDFA = None, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)
        self.safe_specification = safe_specification

    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """

        train_data_file = self._preprocess(X)

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        kwargs.update({'C': '1'})
        success = False
        i_trial = 0


        while not success and i_trial != n_trial:
            try:
                if self.safe_specification:
                    data = self.flexfringe.infer_model(
                        training_file=train_data_file,
                        S=self.safe_specification.graph_data_file,
                        record_time=True, **kwargs)
                else:
                    data = self.flexfringe.infer_model(
                        training_file=train_data_file,
                        record_time=True,
                        **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except:
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class Preprocess(SpecificationEstimator):
    def __init__(self, safe_specification: SafetyDFA = None, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)
        self.safe_specification = safe_specification

    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """

        train_data_file = self._preprocess(X)

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        kwargs.update({'C': '0'}) # C = Choose safetyAlgorithmNum: 0->Preprocessing
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                if self.safe_specification:
                    data = self.flexfringe.infer_model(training_file=train_data_file,
                                                S=self.safe_specification.graph_data_file,
                                                record_time=True,
                                                **kwargs)
                else:
                    data = self.flexfringe.infer_model(training_file=train_data_file,
                                                record_time=True,
                                                **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except Exception as e:
                print(e)
                msg = f'Cannot train a model properly'
                warnings.warn(msg)
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class TargetSpecification(SpecificationEstimator):
    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """
        self._postprocess(X, self.specification)


class HybridSpecification(SpecificationEstimator):
    def __init__(self, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)

    def fit(self, X: Tuple[StateTrajs, ModeTrajs], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """
        Xs, Qs = X

        train_data_file = self._preprocess(Qs, 'train.abbadingo')
        state_file = Automaton.write_traces_to_file(Xs,
            alphabet_size=0,
            file=os.path.join('flexfringe_data', 'states.txt'))

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = self.flexfringe.infer_model(training_file=train_data_file,
                                            Z=state_file,
                                            record_time=True,
                                            **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except Exception as e:
                print(e)
                msg = f'Cannot train a model properly'
                warnings.warn(msg)
                pdfa = None
                i_trial += 1

        self._postprocess(Qs, pdfa)


class HybridSystemEstimator(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, dynamic_matrices):
        self.dynamic_matrices = dynamic_matrices

    @abstractmethod
    def fit(self, X, y=None, **kwargs) -> None:
        pass

    @staticmethod
    def XandQ_to_hybrid_trajs(state_trajectories, mode_trajectories):
        trajs = []
        for X, Q in zip(state_trajectories, mode_trajectories):
            traj = HybridSystemEstimator.XandQ_to_hybrid_traj(X, Q)
            trajs.append(traj)
        return trajs

    @staticmethod
    def XandQ_to_hybrid_traj(X, Q):
        hybrid_traj = []
        for x, q in zip(X, Q):
            hybrid_traj.append([x, q])
        return hybrid_traj

    @staticmethod
    def hybrid_to_XandQ_trajs(hybrid_trajectories):
        state_trajectories = []
        mode_trajectories = []
        for hybrid_traj in zip(hybrid_trajectories):
            X, Q = HybridSystemEstimator.hybrid_to_XandQ_traj(hybrid_traj)
            state_trajectories.append(X)
            mode_trajectories.append(Q)
        return state_trajectories, mode_trajectories

    @staticmethod
    def hybrid_to_XandQ_traj(hybrid_trajectory):
        X = []
        Q = []
        for x, q in hybrid_trajectory:
            X.append(x)
            Q.append(q)
        return X, Q


def find_separating_hyperplane(X1, X2, method: str='svm',
    svm_kwargs={'kernel': 'linear', 'C': 10000}, epsilon: float=0.0,
    plot: bool=True):
    """
    Solve LP Feasibility Problem to find y=[a,b],
    i.e., variables of the hyperplane that separates
    the two non-intersection convex sets X1 and X2

    max or min c, (c\in R)
    s.t.
        a^T * x1 <= b for x1 in X1
        a^T * x2 >= b for x2 in X2
        which can be turned into By>=0
        AND
        1^T * B * y >= 1
    The last inequality makes sure that y!=0 where y=[a,b]^T

    Reference:
    See Problem 4.18 in Stephen Boyd's book "Convex Optimization".
    Solutions can be found here.
    http://egrcc.github.io/docs/math/cvxbook-solutions.pdf
    """
    X1 = np.array(X1)
    X2 = np.array(X2)
    n_data1, n_dim = X1.shape
    n_data2, n_dim = X2.shape
    X = np.r_[X1, X2]
    y = np.r_[-np.ones(n_data1), np.ones(n_data2)]

    if method == 'svm':

        clf = SVC(**svm_kwargs)
        clf.fit(X, y)

        A, b = clf.coef_, clf.intercept_[0]

    elif method == 'lp':

        G1 = np.c_[X1, np.ones(n_data1)] # a^T * X1 + b
        G2 = np.c_[X2, np.ones(n_data2)] # a^T * X2 + b
        G = np.r_[G1, G2]

        # Constraint a constraint s.t. A and b to be nonzero
        H = np.dot(y, G) # y (a^T * X + b)

        model = gpy.Model('lp')
        model.setParam('OutputFlag', False)
        # model.setParam('NodefileStart', 0.5)

        A = []
        variable_names = []
        for i in range(n_dim):
            a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'a{i}')
            A.append(a)
            variable_names.append(f'a{i}')
        b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'b')
        variable_names.append(f'b')

        E = []
        for i in range(n_data1 + n_data2):
            e = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'e{i}')
            variable_names.append(f'e{i}')
            E.append(e)

        for i, g in enumerate(G):
            # y (a^T * x + b) >= 1 - slack
            model.addConstr(y[i] * (g[0]*A[0] + g[1]*A[1] + g[2]*b) >= 1 - E[i])
        # \Sigma_i y_i (a^T * x_i + b) should be none zero to avoid a=0, b=0
        model.addConstr(H[0]*A[0] + H[1]*A[1] + H[2]*b >= 2)

        model.setObjective(sum(E), GRB.MINIMIZE)
        model.optimize()
        status = model.status

        try:
            variables = [model.getVarByName(v).X for v in variable_names[:n_dim+1]]
            slack_variables = [model.getVarByName(v).X for v in variable_names[n_dim+1:]]
        except Exception as e:
            print(e)

        if status == 2 or status==5:
            variables = [model.getVarByName(v).X for v in variable_names[:n_dim+1]]
            slack_variables = [model.getVarByName(v).X for v in variable_names[n_dim+1:]]
            A, b = np.array(variables[:-1]), variables[-1]
        else:
            A, b = None, None
    else:

        raise Exception(f'No such method: {method}')

    if plot:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        lims = plt.axis()
        xmin = np.min(X[:, 0])
        xmax = np.max(X[:, 0])
        xs = np.linspace(xmin - 0.1*abs(xmin), xmax + 0.1*abs(xmax), 100)
        ys = -(A[0]/A[1]) * xs - b/A[1]
        plt.plot(xs, ys, 'r--')
        plt.axis(lims)
        plt.show()

    return A, b


class FitToKModes(HybridSystemEstimator):
    def __init__(self, dynamic_matrices, **kwargs):
        super().__init__(dynamic_matrices, **kwargs)
        self.n_mode = None

    def fit(self, X: Tuple[StateTrajs, ModeTrajs], y=None, **kwargs) -> None:
        self._fit(X, y, **kwargs)

    def _fit(self, X: Tuple[StateTrajs, ModeTrajs], y=None, method='svm', **kwargs) -> None:
        Xs, Qs = X

        valid_modes = self._analyze_modes(Qs)

        # construct a DFA

        # Construct n_mode number of modes
        nodes = {}
        for m in valid_modes:
            nodes[m] = {'is_accepting': False, 'A': self.dynamic_matrices[m]}

        # Separate states at each mode q
        Xtransit = defaultdict(lambda: defaultdict(lambda: []))
        Xstate = defaultdict(lambda: [])
        transitions = defaultdict(lambda: set())
        for X, Q in zip(Xs, Qs):
            prev_q = None
            for i, q in enumerate(Q):
                if prev_q is not None and all([m in valid_modes for m in [prev_q, q]]):
                    transitions[prev_q].add(q)
                    Xtransit[prev_q][q].append(X[i])
                Xstate[q].append(X[i])
                prev_q = q
            last_q = q
            if last_q in valid_modes:
                nodes[last_q]['is_accepting'] = True

        for m in valid_modes:
            nodes[m]['X'] = Xstate[m]

        # find edges excluding self loops
        edges = defaultdict(lambda: defaultdict(lambda: {}))
        for src, targets in transitions.items():
            for target in targets:
                if src == target: continue # self loop

                a, b = find_separating_hyperplane(Xstate[src], Xtransit[src][target], method=method)
                # a, b = find_separating_hyperplane(Xstate[src], Xstate[target], method=method)
                if a is None or b is None:
                    raise Exception('Could not find a separating hyperplane')
                # f = lambda x: np.dot(x, a.T) >= b
                f = lambda x: np.dot(x, a.T) + b >= -1
                edges[src][target]['symbols'] = [f'{a}x>={b}']
                edges[src][target]['guard'] = f
                edges[src][target]['guardA'] = a
                edges[src][target]['guardb'] = b
                edges[src][target]['X'] = Xtransit[src][target]

        (symbol_display_map,
        states,
        edges) = Automaton._convert_states_edges(
            nodes, edges, '$', 'lambda', is_stochastic=False)

        self.dfa = DFA(states, edges,
            symbol_display_map,
            alphabet_size=len(valid_modes),
            num_states=len(nodes),
            start_state=Qs[0][0],
            smooth_transitions=False,
            graph_data_file=None,
            final_transition_sym='$',
            empty_transition_sym='lambda',
        )

    def _analyze_modes(self, Qs):

        valid_modes = set()
        for Q in Qs:
            modes, counts = np.unique(Q, return_counts=True)
            ms = np.squeeze(modes[np.argwhere(counts > 1)])
            if ms.size == 1:
                valid_modes.add(int(ms))
            else:
                for m in ms:
                    valid_modes.add(m)

        valid_modes = set.intersection(set(valid_modes), set(self.dynamic_matrices.keys()))
        valid_modes = list(valid_modes)

        self.n_mode = len(valid_modes)

        return valid_modes

    def get_modes(self):
        if self.n_mode is not None:
            return self.n_mode
        raise Exception('Unknown No. of modes')


class FitToKModesAndMerge(FitToKModes):
    def __init__(self, dynamic_matrices, loss_func=np.linalg.norm, epsilon=3.0,
                 guard_satisfaction_rate=0.1, **kwargs):

        super().__init__(dynamic_matrices, **kwargs)
        self.loss_func = loss_func
        self.epsilon = epsilon
        self.guard_satisfaction_rate = guard_satisfaction_rate

    def fit(self, X, y=None, **kwargs) -> None:
        """
        First Fit to K modes
        Then, merge nodes that are "similar".

        The similarity conditions are:
        1. 'A' matrices must be similar |A' - A| < \epsilon
        2. Set of states must be contained in another set. How? By showing infeasibility.

        Let's try with condition 1 only.
        """
        self._fit(X, y, **kwargs)

        # Find pairs of nodes s.t. A matrices are similar and prepare for merge.
        # Construct a merge relation graph G that directs nodes to the merging nodes
        G = nx.MultiDiGraph()
        for (p, q) in list(combinations(self.dfa.nodes, 2)):
            print(self.loss_func(self.dfa.nodes[p]['A'] - self.dfa.nodes[q]['A']))
            if self.loss_func(self.dfa.nodes[p]['A'] - self.dfa.nodes[q]['A']) < self.epsilon:
                # TODO: For now, the larger (l) set will acquire the smaller (s) set.
                if len(self.dfa.nodes[p]['X']) >= len(self.dfa.nodes[q]['X']):
                    # merge q into p
                    l = p
                    s = q
                else:
                    l = q
                    s = p

                # check if set X in q does not satisfy guards at p (other than the edge to q)
                is_contained_in_l = True
                for k, v in self.dfa[l].items():
                    if k == s: continue
                    Xs = self.dfa.nodes[s]['X']

                    if 'guard' in v[0]:
                        classification_scores = v[0]['guard'](Xs)
                        mean_classification_score = np.mean(classification_scores)
                        if mean_classification_score > self.guard_satisfaction_rate:
                            is_contained_in_l = False

                if is_contained_in_l:
                    G.add_edge(s, l)

        # Find pairs of source (node to be merged) and target (merger node).
        # However, there could be multiple target nodes for the source node.
        # sink_nodes = [node for node, outdegree in G.out_degree(G.nodes()) if outdegree == 0]
        sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

        merge_relations = defaultdict(lambda: [])

        for sink_node in sink_nodes:
            search_queue = queue.Queue()
            search_queue.put(sink_node)

            visited = [sink_node]

            while not search_queue.empty():
                v = search_queue.get()
                for u in G.predecessors(v):
                    if u not in visited:
                        search_queue.put(u)
                        visited.append(u)
                        merge_relations[u].append(sink_node)

        # if there are multiple targets node for the source node, pick larger node.
        for source, targets in merge_relations.items():
            if len(targets) > 1:
                # compare the size of set X, and choose the bigger set.
                ind = np.argmax([len(self.dfa.nodes[n]['X']) for n in targets])
                targets = [ind]

        # Finally, merge source to targets[0]!
        for source, targets in merge_relations.items():

            target = targets[0]
            print(f'Merge Node {source} into Node {target}')
            # 1. merge set X to that of the boss
            self.dfa.nodes[target]['X'] = np.r_[self.dfa.nodes[target]['X'], self.dfa.nodes[source]['X']]

            # 2. delete incoming edges to source
            predecessors = list(self.dfa.predecessors(source))
            for src in predecessors:
                self.dfa.remove_edge(src, source)

            successors = list(self.dfa.successors(source))
            for target in successors:
                self.dfa.remove_edge(source, target)

            # 3. delete state
            self.dfa.remove_node(source)

