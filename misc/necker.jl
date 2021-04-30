### Model code for possible necker cube figure.

@gen (static) function necker()
    ϕ ~ orientation_prior()
    mesh = cube_mesh(orientation)
    binary_pixel_grid = ToPixelGrid(10, 10)(mesh)
    obs ~ Map(FlipBitWithProb(0.1))(binary_pixel_grid)
    return obs
end

@gen (static) function proposal(_)
    ϕ ~ orientation_prior()
    return ϕ
end

inference_circuit = MHCircuit(model, proposal)