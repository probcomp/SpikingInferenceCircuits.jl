"""
    Target

A hardware/software target for information processing.
"""
abstract type Target end

"""
    Spiking <: Target

Spiking circuit target for information processing.
"""
struct Spiking <: Target end

"""
    ComputerProgram <: Target

Target for information processing by running a computer program.
"""
struct ComputerProgram <: Target end