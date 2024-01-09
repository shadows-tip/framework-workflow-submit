def get_initial_data(Nd=40000):
    Nh = 1400000  # the population of Yangquan, Shanxi is 1.4 million
    Ns = 100 * Nh  # the number of sandflies
    Nd = Nd

    Eh0, Ih0, Rh0 = 50, 1, 10  # the initial value of a person
    Ed0, Id0, Rd0 = 2000, 500, 600  # the initial value of the dog
    Es0, Is0 = 6000, 50  # the initial value of sandflies

    Sh0 = Nh - Eh0 - Ih0 - Rh0
    Ss0 = Ns - Es0 - Is0
    Sd0 = Nd - Ed0 - Id0 - Rd0

    initial_values = [Sh0, Eh0, Ih0, Rh0, Sd0, Ed0, Id0, Rd0, Ss0, Es0, Is0, Nh, Nd, Ns]
    return initial_values


def get_fixed_param():
    miuh = 0.0000367  # Natural mortality rate in humans
    alphah = 0.00631  # Specific lethality of leishmaniasis in humans
    bh = 0.01  # Probability of infection after being bitten by sandfly in humans
    deltah = 0.011  # Latent recovery rate in humans
    faih = 0.0056  # Inverse of incubation period in humans (1/180)
    rouh = 0.02  # Recovery rate from clinically ill to immune in humans

    miud = 0.000228  # Natural mortality rate in dogs
    alphad = 0.00181  # Specific lethality of leishmaniasis in dogs
    bd = 0.01  # Probability of infection after being bitten by sandfly in dogs
    deltad = 0.00822  # Latent recovery rate in dogs
    faid = 0.0111  # Inverse of incubation period in dogs (1/90)
    roud = 0.000904  # Recovery rate from clinically ill to immune in dogs
    mius = 0.05  # Natural mortality rate of sandflies
    cy = 0.247  # Probability of clinically ill dog infecting a sandfly
    tao = 7  # Incubation period of pathogen in the sandfly
    ch = 0.0531  # Number of humans bitten by infected sandflies in a day
    cd = 15 * ch  # Number of dogs bitten by infected sandflies in a day
    fixed_param = [miuh, alphah, bh, deltah, faih, rouh, miud, alphad, bd, deltad, faid, roud, mius, cy, tao, ch, cd]
    return fixed_param
