===============================
Getting Started with MBNpy
===============================

This guide intorudces using MBNpy for **risk/reliability analysis of system event**.
This guide walks through **installation, defining variables and probability distributions, and performing system analysis**.

.. contents:: Table of Contents
   :depth: 2
   :local:

---------------------------------

Installation
------------

MBNpy is available via **pip**:

.. code-block:: bash

   pip install mbnpy

To install from source:

.. code-block:: bash

   git clone https://github.com/jieunbyun/MBNpy.git
   cd MBNpy
   pip install .

MBNpy requires **Python 3.12+**.

---------------------------------

Example Problem
---------------
This example demonstrates application for the **reliability block diagram** from the `2019 MBN paper <https://doi.org/10.1016/j.ress.2019.01.007>`_.

The system consists of **8 components** (:math:`X1` to :math:`X8`), each with **two states**:

- :math:`0`: Failure

- :math:`1`: Survival
Components are **statistically independent** with:

- :math:`P(X_i=0) = 0.1`

- :math:`P(X_i=1) = 0.9`
The **system state (:math:`X_9`) depends on the connection from source to sink**.

.. figure:: _static/img/rbd_ex/rbd.jpg
   :width: 600px
   :align: center
   :alt: Reliability Block Diagram

   **Figure 1**: Reliability block diagram (RBD).

The **Bayesian network (BN) representation** is:

.. figure:: _static/img/rbd_ex/rbd_bn.jpg
   :width: 300px
   :align: center
   :alt: Bayesian Network Representation

   **Figure 2**: Bayesian network representation of the system.

---------------------------------

Objectives
---------------
1. Compute the **system failure probability** :math:`P(X_9=0)`.

2. Identify **critical components** using component importance measures.

For technical details, see:
   
   **Byun et al. (2019)**  
   *Matrix-based Bayesian networks, Reliability Engineering & System Safety, 185, 533-545.*

   **Byun & Song (2021)**  
   *Generalized matrix-based Bayesian networks, Reliability Engineering & System Safety, 211, 107468.*

---------------------------------

Defining Variables in MBNpy
---------------------------
To model a Bayesian network, **variables must be defined first**.

Each **component** is a `Variable` object:

.. code-block:: python

   from mbnpy import variable

   varis = {}
   varis['x1'] = variable.Variable(name='x1', values=['f', 's'])

- The **variable dictionary** (``varis``) stores all model variables.
- ``"f"`` and ``"s"`` correspond to **failure** (``0``) and **survival** (``1``).
- The numerical index assigned to each state is determined by its position in the ``values`` list.

All 8 components are defined using a loop:

.. code-block:: python

   n_comp = 8
   for i in range(1, n_comp):
       varis[f'x{i+1}'] = variable.Variable(name=f'x{i+1}', values=['f', 's'])

   print(varis['x8'])  # Check last component

The system variable (:math:`X_9`) is defined similarly:

.. code-block:: python

   varis['x9'] = variable.Variable(name='x9', values=['f', 's'])

---------------------------------

Defining Probability Distributions
---------------------------------
The **conditional probability matrix (CPM)** represents probability distributions.

Since **components** (:math:`X_1` to :math:`X_8`) **have no parents**, they are defined as **marginal probabilities**:

.. code-block:: python

   from mbnpy import cpm
   import numpy as np

   cpms = {}
   cpms['x1'] = cpm.Cpm(
       variables=[varis['x1']], 
       no_child=1, 
       C=np.array([[0], [1]]), 
       p=np.array([0.1, 0.9])
   )

- ``variables``: A list of variables involved in the distribution (e.g., ``X1``).
- ``no_child``: The number of child nodes (1 in this case).
- ``C`` (Event Matrix): Specifies the state of each instance.
- ``p`` (Probability Vector): Defines the corresponding probabilities.

This process is repeated for all components:

.. code-block:: python

   for i in range(1, n_comp):
       cpms[f'x{i+1}'] = cpm.Cpm(
           variables=[varis[f'x{i+1}']], 
           no_child=1, 
           C=np.array([[0], [1]]), 
           p=np.array([0.1, 0.9])
       )

The **system** (:math:`X_9`) **is defined using the branch-and-bound method**:

.. figure:: _static/img/rbd_ex/rbd_bnb.jpg
   :width: 400px
   :align: center

The ``Csys`` and ``psys`` matrices are constructed:

.. code-block:: python

   Csys = np.array([
       [0, 2, 2, 2, 2, 2, 2, 2, 0],
       [0, 2, 2, 2, 2, 2, 2, 0, 1],
       [1, 1, 2, 2, 2, 2, 2, 1, 1],
       [1, 0, 1, 2, 2, 2, 2, 1, 1],
       [1, 0, 0, 1, 2, 2, 2, 1, 1],
       [0, 0, 0, 0, 0, 2, 2, 1, 1],
       [0, 0, 0, 0, 1, 0, 2, 1, 1],
       [0, 0, 0, 0, 1, 1, 0, 1, 1],
       [1, 0, 0, 0, 1, 1, 1, 1, 1]
   ])
   psys = np.array([[1.0] * 9]).T

   cpms['x9'] = cpm.Cpm(
       variables=[varis['x9'], varis['x1'], varis['x2'], varis['x3'], varis['x4'],
                  varis['x5'], varis['x6'], varis['x7'], varis['x8']],
       no_child=1,
       C=Csys,
       p=psys
   )

---------------------------------

System Analysis
---------------
Variable elimination is applied to compute the system failure probability. To determine the probability distribution of the system event, all component variables except for :math:`X_9` are eliminated.

.. code-block:: python

   from mbnpy import inference

   cpm_sys = inference.variable_elim(
       cpms, [varis[f'x{i+1}'] for i in range(8)]
   )

   print(f'System failure probability: {cpm_sys.p[0][0]:.2f}')

Component importance measures are calculated, following `Kang et al. (2008) <https://doi.org/10.1016/j.ress.2008.02.011>`_'s definition :math:`CIM_i = P(X_i=0|X_9=0) = P(X_i=0,X_9=0) / P(X_9=0)`:

.. code-block:: python

   CIMs = {}
   for i in range(n_comp):
       varis_elim = [varis[f'x{j+1}'] for j in range(n_comp) if j != i]
       cpm_sys_xi = inference.variable_elim(cpms, varis_elim)
       prob_s0_x0 = cpm_sys_xi.get_prob([f'x{i+1}', 'x9'], [0,0])
       CIMs[f'x{i+1}'] = prob_s0_x0 / cpm_sys.p[0][0]

   print(CIMs)
