declare name "Ghost Strings";
declare description "Ethereal bowed-string style detuned saws.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 8, 1, 24, 0.1);
cutoff = hslider("cutoff", 2600, 400, 9000, 1);

envelope = gain * en.adsr(0.3, 0.55, 0.78, 1.1, gate);
voice = os.sawtooth(freq * (1 + spread / 1800)) + os.sawtooth(freq * (1 - spread / 1800));
process = voice : fi.lowpass(2, cutoff) * envelope * 0.55 <: _, _;
