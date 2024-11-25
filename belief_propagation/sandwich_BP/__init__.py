"""
Tools for simulating MPS time evolution and
the ground state problem using MPS and PEPOs.

I consider a typical workflow to be:
* Define a state as an instance of the `MPO` class,
* define operators as instances of (subclasses of) the `PEPO` class,
* and do stuff with them.
"""