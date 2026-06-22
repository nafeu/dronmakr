declare name "Choir Cloud";
declare description "Stacked detuned sines for a vocal cloud.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 6, 1, 20, 0.1);

envelope = gain * en.adsr(0.25, 0.5, 0.8, 1.0, gate);
voice = os.osc(freq * (1 + spread / 2000))
      + os.osc(freq)
      + os.osc(freq * (1 - spread / 2000));
process = voice * 0.33 * envelope <: _, _;
