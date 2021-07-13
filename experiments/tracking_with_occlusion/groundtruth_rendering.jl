### Deterministic ground-truth trace rendering.
pixel_color_determ((occ, sqx, sqy), (x, y)) =
    is_in_square(sqx, sqy, x, y) ? 2 :
    is_occluded(occ, x, y)       ? 1 :
                                    0

image_determ(occ, sqx, sqy) = [
    pixel_color_determ((occ, sqx, sqy), (x, y))
    for x=1:ImageSideLength(), y=1:ImageSideLength()
]

image_determ(cm) = image_determ(cm[:occ => :val], cm[:x => :val], cm[:y => :val])
image_determ(cm::Gen.ChoiceMap, t) = image_determ(get_submap(cm, t))
image_determ(tr::Gen.Trace, t) = image_determ(get_choices(tr), t)