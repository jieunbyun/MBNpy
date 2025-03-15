======================
Using the `variable` Module
======================

The `variable` module in MBNpy provides a class for managing **discrete-state system variables**, offering tools for state representation, mapping, and transformation.

This guide explains how to:
- Create and manipulate `Variable` objects
- Access and modify variable properties
- Use mapping functions for state representation

-----------------------

Setting Up
----------

To use the `variable` module, first import it:
   
.. code-block:: python

   from mbnpy import variable

Creating a Variable
-------------------

A `Variable` object represents a **discrete-state variable** with:
- A **name** (string)
- A list of **possible states**
- An optional **mapping matrix (`B`)** for composite states

.. code-block:: python

   my_var = variable.Variable(name="X", values=["Low", "Medium", "High"])

Accessing Variable Properties
-----------------------------

Each `Variable` object has the following key attributes:

- **`name`** → The name of the variable
- **`values`** → The list of possible states
- **`B`** → The mapping matrix (if applicable)
- **`B_flag`** → Controls how `B` is generated

Example:

.. code-block:: python

   print(my_var.name)   # Output: "X"
   print(my_var.values) # Output: ["Low", "Medium", "High"]
   print(my_var.B_flag) # Output: "auto"

Modifying a Variable
--------------------

You can modify a variable's **name**, **values**, or **mapping behavior**.

Changing the name:
   
.. code-block:: python

   my_var.name = "New_X"
   print(my_var.name)  # Output: "New_X"

Updating the state values:

.. code-block:: python

   my_var.values = ["Failure", "Survival"]
   print(my_var.values)  # Output: ["Failure", "Survival"]

Defining the Mapping Matrix
---------------------------

The **mapping matrix (`B`)** defines how composite states relate to basic states.

By default:
- If the variable has ≤ 6 states, `B` is automatically generated.
- If the variable has > 6 states, `B` is not stored due to memory constraints.

You can **manually control `B` storage** using the `B_flag` parameter:
   
.. code-block:: python

   var1 = variable.Variable("X", ["Low", "Medium", "High"], B_flag="auto")  # Default: auto mode
   var2 = variable.Variable("Y", ["Yes", "No"], B_flag="store")  # Force storing B
   var3 = variable.Variable("Z", ["A", "B", "C", "D"], B_flag="fly")  # No precomputed B

Getting the Mapping Matrix (`B`)
------------------------------

To retrieve the **mapping matrix**:

.. code-block:: python

   print(var1.B)   # Outputs: [{0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}]
   print(var2.B)   # Outputs: [{0}, {1}]
   print(var3.B)   # Outputs: None (since B_flag="fly")

-----------------------

Finding State Index from a Set
------------------------------

To find the **state index** of a given set of basic states:

.. code-block:: python

   my_var = variable.Variable(name="X", values=[0, 1, 2])
   state_idx = my_var.get_state({0, 1})  
   print(state_idx)  # Output: 3

-----------------------

Finding the Set from a State Index
----------------------------------

To retrieve the **basic states** corresponding to a given **state index**:

.. code-block:: python

   state_set = my_var.get_set(3)
   print(state_set)  # Output: {0,1}

-----------------------

Converting a Binary Vector to State Index
-----------------------------------------

The function `get_state_from_vector` finds a **state index** given a **binary vector** representation of states.

Example:

.. code-block:: python

   binary_vector = [1, 0, 1]  # Represents states {0,2}
   state_index = my_var.get_state_from_vector(binary_vector)
   print(state_index)  # Output: 4

If all values in the vector are 0, the function returns `-1`:

.. code-block:: python

   binary_vector = [0, 0, 0]  
   state_index = my_var.get_state_from_vector(binary_vector)
   print(state_index)  # Output: -1

-----------------------

Using `Bvec` to Get State Indices
---------------------------------

`get_Bst_from_Bvec` converts a **3D binary matrix (`Bvec`)** into **state indices**.

Example:

.. code-block:: python

   Bvec = np.array([
       [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
       [[0, 1, 1], [0, 0, 0], [0, 1, 0]]
   ])

   Bst = my_var.get_Bst_from_Bvec(Bvec)
   print(Bst)

-----------------------

Testing `variable.py`
---------------------

The module can be tested using **pytest**.

Run all tests:
```cmd
pytest tests/test_variable.py
