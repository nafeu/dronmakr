declare name "Octave Fifth";
declare description "Root, octave, and fifth blend for power chords.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.05, 0.22, 0.84, 0.4, gate);
voice = os.sawtooth(freq) + 0.6*os.sawtooth(freq*2) + 0.45*os.sawtooth(freq*1.5);
process = voice * 0.38 * envelope <: _, _;
