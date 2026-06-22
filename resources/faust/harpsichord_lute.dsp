declare name "Harpsichord Lute";
declare description "Plucked lute-harpsichord hybrid tone.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.001, 0.35, gate);
process = os.square(freq) : fi.lowpass(1, 4200) * envelope * 0.55 <: _, _;
