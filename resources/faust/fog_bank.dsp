declare name "Fog Bank";
declare description "Wide soft noise fog with slow motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.25, 0.65, 0.6, 1.2, gate);
process = no.noise : fi.lowpass(2, 1100) * envelope * (0.75 + 0.25*os.lf_triangle(0.11)) * 0.45 <: _, _;
