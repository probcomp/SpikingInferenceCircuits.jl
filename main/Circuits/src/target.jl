"""
    Target

A hardware/software target for information processing.
"""
abstract type Target end

# Eg. could have something like:
# """
#     ComputerProgram <: Target

# Target for information processing by running a computer program.
# """
# struct ComputerProgram <: Target end