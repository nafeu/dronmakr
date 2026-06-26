import("stdfaust.lib");

bits = hslider("bits", 10, 4, 16, 1);
mix = hslider("mix", 0.38, 0, 1, 0.01);
steps = pow(2.0, bits);
wet(x) = x * steps : round : /(steps);
process = wet * mix + _ * (1 - mix);
