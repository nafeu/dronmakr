declare name "Iron Saw";
declare description "Hard clipped saw stack for industrial tones.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.02, 0.14, 0.75, 0.28, gate);
voice = os.sawtooth(freq) + 0.4*os.sawtooth(freq*0.997);
process = voice * 0.48 * envelope <: _, _;
