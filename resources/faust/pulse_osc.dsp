declare name "Pulse Oscillator";
declare description "Variable-width pulse wave with a short ADSR envelope.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
width = hslider("width", 0.5, 0.05, 0.95, 0.01);

envelope = gain * en.adsr(0.05, 0.12, 0.88, 0.45, gate);
process = (os.osc(freq) > (width * 2 - 1)) * envelope <: _, _;
