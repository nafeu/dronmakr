declare name "Sub Stack";
declare description "Saw layered with a sub-octave for heavy low-end.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
sub = hslider("sub", 0.65, 0, 1, 0.01);

envelope = gain * en.adsr(0.05, 0.2, 0.85, 0.4, gate);
voice = os.sawtooth(freq) + os.sawtooth(freq * 0.5) * sub;
process = voice * envelope <: _, _;
