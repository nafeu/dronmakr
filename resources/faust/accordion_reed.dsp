declare name "Accordion Reed";
declare description "Wheezy accordion reed with detune.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.04, 0.12, 0.82, 0.28, gate);
voice = os.square(freq*1.004) + os.triangle(freq*0.996);
process = voice * 0.45 * envelope <: _, _;
