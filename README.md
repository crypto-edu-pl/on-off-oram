## oram ![Build Status](https://github.com/facebook/oram/workflows/CI/badge.svg)

This library implements an Oblivious RAM (ORAM) for secure enclave applications.

Oblivious RAM (ORAM) is a technique that allows a client to fully hide its pattern of accesses to memory stored by an untrusted third party, such as a cloud server. ORAM is costly, with a proven asymptotically logarithmic overhead that is at least \(10 \times\) to \(100 \times\) in practice. However, in some applications only some of the accesses to the untrusted memory may be sensitive. In this work, we introduce ON-OFF ORAM: an extension to ORAM schemes that allows the client to avoid the unnecessary overhead of protecting non-sensitive accesses by switching between two modes: ON, in which the client's memory accesses are oblivious just like in regular ORAM, and OFF, in which they are not. We implement ON-OFF Path ORAM---an application of the ON-OFF extension to Path ORAM, suitable for protecting the memory accesses of enclaves---and show performance improvements both in online and total overhead.

This crate assumes that ORAM clients are running inside a secure enclave architecture that provides memory encryption.
It does not perform encryption-on-write and thus is **not** secure without memory encryption.

⚠️ **Warning**: This implementation has not been audited. Use at your own risk!

Documentation
-------------

------------


### Minimum Supported Rust Version

Rust **1.81** or higher.

Resources
---------

- [Original Path ORAM paper](https://eprint.iacr.org/2013/280.pdf), which introduced the standard "vanilla" variant of Path ORAM on which this library is based.
- [Path ORAM retrospective paper](http://elaineshi.com/docs/pathoram-retro.pdf), containing a high-level overview of developments related to Path ORAM.
- [Oblix paper](https://people.eecs.berkeley.edu/~raluca/oblix.pdf), which describes the oblivious stash data structure this library implements. 

Contributors
------------

The authors of this code are Woiciech Wisniewski ([@wciszewski] and Emanuele Ragnoli[@u2135]

Code Organization
--------------------

License
-------

This project is dual-licensed under either the [MIT license](https://github/crypto-edu-pl/on-off-oram/main/LICENSE-MIT)
or the [Apache License, Version 2.0](https://github/crypto-edu-pl/on-off-oram/main/LICENSE-APACHE).
You may select, at your option, one of the above-listed licenses.

