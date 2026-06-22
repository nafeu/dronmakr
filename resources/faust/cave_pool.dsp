declare name "Cave Pool";
declare description "Dark cave pool with dripping noise hints.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.55, 0.9, 0.62, 2.0, gate);
tone = os.triangle(freq*0.5) * 0.55;
drip = no.noise : fi.highpass(1, 2000) * 0.08;
process = (tone + drip) * envelope <: _, _;
