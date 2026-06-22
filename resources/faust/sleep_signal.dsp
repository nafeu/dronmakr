declare name "Sleep Signal";
declare description "Hypnotic low pulse bed for deep drones.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.6, 1.0, 0.6, 2.3, gate);
bed = os.osc(freq*0.5) * (0.7 + 0.3*os.lf_triangle(0.06));
process = bed * envelope <: _, _;
