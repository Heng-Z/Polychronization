import brainpy as bp
bp.math.set_platform('cpu')

E = bp.dyn.Izhikevich(80, a=0.02, b=0.2, c=-65, d=8)
I = bp.dyn.Izhikevich(20, a=0.1, b=0.2, c=-65, d=2)
E.v[:] = -65
E.u[:] = -16
I.v[:] = -65
I.u[:] = -14

E2E = bp.connect.FixedProb(prob=0.1)