declare name "Steam Hiss";
declare description "Airy steam hiss through a moving band-pass.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
cut = hslider("cutoff", 1600, 400, 5000, 1);
envelope = gain * en.adsr(0.15, 0.45, 0.62, 0.9, gate);
wander = cut * (1 + 0.25 * os.lf_triangle(0.22));
process = no.noise : fi.resonbp(2, wander, 2) * envelope * 0.42 <: _, _;
