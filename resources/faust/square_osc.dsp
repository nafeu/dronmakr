declare name "Square Oscillator";
declare description "Hollow square wave with a short ADSR envelope.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");

envelope = gain * en.adsr(0.05, 0.12, 0.88, 0.45, gate);
process = os.square(freq) * envelope <: _, _;
