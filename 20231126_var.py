a = [0.018060663704217973, 0.020907298586808537, 0.02506906266895781, 0.02520452627689994, 0.02574810711092941, 0.025798397277157686, 0.027379708407642707, 0.02939248077370911, 0.030181036026448192, 0.031126879895079194, 0.034256783178229853, 0.03486586324386873, 0.03554834110128909, 0.03556501022956489, 0.035596377556322764, 0.03581358906136671, 0.036158421796969265, 0.03623451915704521, 0.036353150920585504, 0.03808262483390789, 0.03973904148464344, 0.03988769157634397, 0.0405788552141287, 0.04057928842790857, 0.04078567422351499, 0.041015562433519, 0.041149267872319636, 0.04178668377073309, 0.04338479146892318, 0.04351685123397075, 0.043839561006042665, 0.04429872107608135, 0.04431361042966235, 0.04488322147089018, 0.045088546247434885, 0.04520110082565483, 0.04545196460449868, 0.0458487731242177, 0.04592166172079064, 0.0462612968026716, 0.04642741546561772, 0.0468749863806429, 0.04717385550590292, 0.04752684695486315, 0.04787866590646323, 0.04801661179340395, 0.0487638123235338, 0.04914925914664873, 0.04943786202415068, 0.049721008306279874, 0.04975447299035374, 0.04975995046610121, 0.049903394119216356, 0.050729321072073175, 0.05084243558016532, 0.051382596902818115, 0.05152345235082429, 0.05194863330830436, 0.05195884548952249, 0.052505839944068086, 0.05251561107376292, 0.053009727792350615, 0.053074232091783635, 0.05373877079517119, 0.05390200329988924, 0.0539227687657076, 0.05397607654911258, 0.054066443953509104, 0.05468073659246053, 0.05524941426618587, 0.05576451541999835, 0.05629960549167305, 0.05648657592311621, 0.05648986044457403, 0.057294596744335756, 0.05783892312695087, 0.05853055554367025, 0.05904477253041737, 0.05913968339413282, 0.059457503349124405, 0.060348023728445555, 0.060468712537065865, 0.062373632938725464, 0.06381526531793288, 
0.06544049245322756, 0.06585032680425337, 0.06631419454389828, 0.06645380884279097, 0.066828855177605, 0.06693462230789783, 0.06734409740013145, 0.07108273880917536, 0.07154401325802956, 0.07348921705814053, 0.07407022628528215, 0.07416290739521907, 0.07662113282518174, 0.07664124464289204, 0.0817601298234858, 0.08396326378005349]

b = [0]*len(a)

for i in range(len(a)): 
    b[i] = round(a[i], 5)

print((b[-5:]))
print((sum(b[-5:])/5))