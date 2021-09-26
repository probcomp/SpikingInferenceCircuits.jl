### Deterministic ground-truth trace rendering.
pixel_color_determ((occ, sqx, sqy), (x, y)) =
    _is_in_square(sqx, sqy, x, y) ? 2 :
    _is_occluded(occ, x)          ? 1 :
                                    0

image_determ(occ, sqx, sqy) = [
    pixel_color_determ((occ, sqx, sqy), (x, y))
    for x=1:ImageSideLength(), y=1:ImageSideLength()
]
img_determ_with_colors(args...) = [PixelColors()[color_idx + 1] for color_idx in image_determ(args...)]

image_determ(cm) = image_determ(cm[:occ => :val], cm[:x => :val], cm[:y => :val])
image_determ(cm::Gen.ChoiceMap, t) = image_determ(get_submap(cm, t))
image_determ(tr::Gen.Trace, t) = image_determ(get_choices(tr), t)