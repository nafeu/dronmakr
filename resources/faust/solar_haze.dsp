declare name "Solar Haze";
declare description "Wide detuned pad with a soft noise halo.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 10, 1, 30, 0.1);

envelope = gain * en.adsr(0.45, 0.7, 0.68, 1.6, gate);
pad = os.triangle(freq * (1 + spread / 1500)) + os.triangle(freq * (1 - spread / 1500));
halo = no.noise : fi.lowpass(1, 1800) * 0.12;
process = (pad * 0.45 + halo) * envelope <: _, _;
