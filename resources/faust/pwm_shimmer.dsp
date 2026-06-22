declare name "PWM Shimmer";
declare description "Pulse wave with slow width modulation.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
lfoRate = hslider("lfo", 0.35, 0.05, 4, 0.01);

envelope = gain * en.adsr(0.05, 0.15, 0.85, 0.35, gate);
width = 0.5 + 0.35 * os.lf_triangle(lfoRate);
process = (os.osc(freq) > (width * 2 - 1)) * envelope <: _, _;
