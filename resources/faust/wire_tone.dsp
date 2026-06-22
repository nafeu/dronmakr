declare name "Wire Tone";
declare description "Thin, tense high square with bite.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.02, 0.12, 0.78, 0.25, gate);
process = os.square(freq) * envelope * 0.55 <: _, _;
