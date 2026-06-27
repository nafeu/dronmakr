import("stdfaust.lib");

size = hslider("size", 10, 2, 20, 0.1);
damp = hslider("damp", 0.45, 0, 1, 0.01);
mix = hslider("mix", 0.75, 0, 1, 0.01);

process(l, r) = l * (1 - mix) + wL * mix, r * (1 - mix) + wR * mix
with {
    mid = (l + r) * 0.5;
    wet = mid, mid : re.greyhole(size, damp, 0.98, 0.48, 0.96, 3200, 1);
    wL = wet : _, !;
    wR = wet : !, _;
};
