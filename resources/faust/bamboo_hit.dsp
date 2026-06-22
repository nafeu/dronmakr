declare name "Bamboo Hit";
declare description "Dry bamboo knock with a woody tone.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.001, 0.28, gate);
process = os.triangle(freq) : fi.lowpass(1, 3200) * envelope * 0.75 <: _, _;
