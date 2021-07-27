@gen (static) function step_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, imgₜ)
    occ ~ occ_pos_dist(occₜ₋₁, imgₜ)
    expected_x = truncate(xₜ₋₁ + vxₜ₋₁, positions(SquareSideLength()))

    # TODO: do this with nested `Map`
    square_should_overlap = [
        occ_overlaps(pixx, occ) ? :indifferent :
        imgₜ[pixx, pixy] ? :yes : :no
        for pixx=1:ImageSideLength(),
            pixy=1:ImageSideLength()
    ]

    x ~ SumProductDist(
        possible_values=positions(SquareSideLength()),
        sum_over=positions(SquareSideLength()),
        product_terms = [
            (x, y, expected_x) -> movement_dist(expected_x)[x],
            [ # unpack array
                [ # unpack array
                    (x, y, square_should_overlap_here, pixx, pixy) ->
                        square_should_overlap_here == :indifferent ? 1. :
                        (
                            (square_should_overlap_here == :yes && square_covers(x, y, pixx, pixy)) ||
                            (square_should_overlap_here == :no && !square_covers(x, y, pixx, pixy))
                        ) ? 0.9 : 0.1
                ]
            ]
        ]
    )(expected_x, square_should_overlap)

    expected_y = truncate(yₜ₋₁ + vyₜ₋₁, positions(SquareSideLength()))
    y_should_overlap = [
        square_should_overlap[x, pixy]
        for pixy=1:ImageSideLength()
    ]

    y ~ ProductDist(
        possible_values=positions(SquareSideLength()),
        product_terms = [
            (y, expected_y) -> movement_dist(expected_y)[y],
            (y, y_should_overlap) ->
                y_should_overlap == :indifferent ? 1. :
                (
                    (y_should_overlap == :yes && square_side_covers(y, pixy)) ||
                    (y_should_overlap == :no && !square_side_covers(y, pixy))
                ) ? 0.9 : 0.1
        ]
    )(expected_y, y_should_overlap)

    expected_vx = truncate(x - xₜ₋₁, Vels())
    expected_vy = truncate(y - yₜ₋₁, Vels())
    vx ~ vel_step_proposal_dist(vxₜ₋₁, expected_vx)
    vy ~ vel_step_proposal_dist(vyₜ₋₁, expected_vy)
end
vel_step_proposal_dist = ProductDist(
    possible_values=Vels(),
    product_terms=[
        (v, prev_v) -> prob_v_transition(prev_v)[v],
        # TODO: be exact with the below so it is the same as the probability of having gotten this transition given the velocity
        (v, expected_v_from_pos) -> truncated_discretized_gaussian(expected_v_from_pos, 2.)
    ]
)



#=
First draft:
    x ~ SumProductDist(
        possible_values=positions(SquareSideLength()),
        sum_over=positions(SquareSideLength()),

        # TODO: probably don't need to put in dependent_types?  since it can be inferred?
        dependent_types=( # types of inputs, and their groupings
            (positions(SquareSideLength())),
            (Product(Product([:indifferent, :yes, :no])))
        ),

        (x, y, (expected_x,), (pixels_expecting_square_overlap,)) ->
        # first = sampled val; second = summed over val; rest = passed in args
            [ # terms to multiply together
                x_move_dist(expected_x)[x],
                (
                    pixels_expecting_square_overlap[pixx][pixy] == :indifferent ? 1. :
                    (
                        (pixels_expecting_square_overlap[pixx][pixy] == :yes &&
                        square_over(x, y, pixx, pixy))
                        || (pixels_expecting_square_overlap[pixx][pixy] == :no &&
                        !square_over(x, y, pixx, pixy))
                    ) ? 0.9 : 0.1
                    for pixx=1:ImageSideLength(),
                        pixy=1:ImageSideLength()
                )...
            ]
    )((expected_x,), (occ, imgₜ))

=#