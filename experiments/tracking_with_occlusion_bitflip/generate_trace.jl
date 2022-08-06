function generate_trace(occs, xs, ys, vxs, vys) # TODO: support giving the observations
    zipped = zip(occs, xs, ys, vxs, vys)
    ((occ, x, y, vx, vy), rest) = Iterators.peel(zipped)
    nsteps = length(collect(rest))
    tr, _ = generate(model, (nsteps,), choicemap(
        (:init => :latents => :occₜ => :val, occ),
        (:init => :latents => :xₜ => :val, x),
        (:init => :latents => :yₜ => :val, y),
        (:init => :latents => :vxₜ => :val, vx),
        (:init => :latents => :vyₜ => :val, vy),
        Iterators.flatten(
            (
                (:steps => t => :latents => :occₜ => :val, occ),
                (:steps => t => :latents => :xₜ => :val, x),
                (:steps => t => :latents => :yₜ => :val, y),
                (:steps => t => :latents => :vxₜ => :val, vx),
                (:steps => t => :latents => :vyₜ => :val, vy)
            )
            for (t, (occ, x, y, vx, vy)) in enumerate(rest)
        )...
    ))
    return tr
end

### Script checking to confirm that trace generated produces deterministic observation: ###
matrix_to_vec_of_vecs(matrix) = [reshape(matrix[:, x], (:,)) for x=1:size(matrix)[2]] # matrix is indexed [y, x]
occ = [2, 2, 2]; x = [1, 1, 1]; y = [3, 2, 1]; vx = [0, 0, 0]; vy = [-1, -1, -1]
det_rendered_imgs = [(matrix_to_vec_of_vecs(image_determ(args...))) for args in zip(occ, x, y)]

to_v_of_v(cm) =
    [
        cm[:img_inner => x => y => :got_photon => :val] ? 1 : 0
        for x=1:3, y=1:3
    ]

tr = generate_trace(occ, x, y, vx, vy)
obs_cms = [
    get_submap(get_choices(tr), :init => :obs),
    (
        get_submap(get_choices(tr), :steps => t => :obs)
        for t=1:get_args(tr)[1]
    )...
]

for (img, obs_cm) in zip(det_rendered_imgs, obs_cms)
    for x=1:length(img)
        for y=1:length(img[x])
            if (img[x][y] != 0) != obs_cm[:img_inner => x => y => :got_photon => :val]
                error("$x, $y, $(img[x][y] != 0), $(obs_cm[:img_inner => x => y => :got_photon => :val])")
            end
        end
    end
end

