declare name "Thumb Piano";
declare description "Warm thumb piano pluck with body.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.002, 0.75, gate);
process = os.osc(freq) : fi.lowpass(1, 4200) * envelope <: _, _;
