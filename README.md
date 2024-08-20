# Repository-Name

    An implementation of the NPA hierarchy using python using the cvxpy library for SDP solutions.

## Requirements

    The requirements for this repository are as follows:

    - Python 3.6 or higher
    - NumPy library
    - cvxpy

    Please make sure you have these dependencies installed before running the code.

## Usage

    To use the code, please see the jupyter notebook npa.ipynb for an example.
    1) One must specify the Bell scenario in hand:
        * NX - Number of possible inputs for party A.
        * NA - Number of possible outputs for party A.

        * NM - Number of possible inputs for party B.
        * NB - Number of possible outputs for party B.
    2) Specify a distribution (there are 3 examples for distributions).
    3) Specify the depth of the hierarchy.
    3) Run the code! - A negative outcome means that the distribution is not quantumly feasible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.