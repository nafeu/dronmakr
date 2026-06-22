declare name "Guitar Mute";
declare description "Muted guitar pluck with short body.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.001, 0.22, gate);
process = os.sawtooth(freq) : fi.lowpass(1, 2600) * envelope * 0.6 <: _, _;
