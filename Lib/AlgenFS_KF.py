
import os
import sys
dirX = os.getcwd()
print(dirX)
sys.path.append(dirX+"/lib")
import Algen_lib as lib
import numpy as np


def features_selection(data, label, jumlah_populasi, jumlah_titik, prob_mutasi,K, alpha, target, 
                        panjang_fitur, fitur, metode='tf',additional_child=True):
    features_1_bin = [1 for i in range(panjang_fitur)]
    pop = lib.create_population(jumlah_populasi-1, panjang_fitur)
    pop.append(features_1_bin)
    # print(sum())
    fns_1 = lib.fitness_kf(data, label, features_1_bin, fitur, alpha = alpha, metode = metode, K=K)
    print("fitness all features:",fns_1)

    fitnes_pop = list()
    # temp_features_fitness = list()
    for features_bin in pop:
        fns = lib.fitness_kf(data, label, features_bin, fitur, alpha = alpha, metode = metode, K=K)
        fitnes_pop.append(fns)

    print(len(fitnes_pop),"|",len(pop))

    pop_used = dict()
    popp_0 = list(pop)
    for p, fit in zip(popp_0, fitnes_pop):
        bin_str = lib.int_to_str(p) 
        pop_used.update({bin_str:fit})

    # print("")

    # print(pop[0])

    best_fitur_list = list()
    best_fitness_list = list()
    best_generasi_list = list()

    best_jumlah_fitur_list = list()

    gen=0
    while True:
        gen+=1
        print("-"*25)
        print("Generasi ke",gen , "[Terbaik]")
        print("   "+"___________________________________________")
        print("     Nilai Fitness       |Fitur|ALL  |Generasi")
        rw = lib.get_roulette_wheel_(fitnes_pop, inc=jumlah_populasi)
        parents = lib.select_parents(rw)

        best_fitur_list_per_generasi = list()
        best_fitness_list_per_generasi = list()
        for p in parents:
            best_generasi_list.append(gen)
            family_fitness=list()
            family_fitur=list()

            mama_index = p[0]
            papa_index = p[1]

            m4ms = fitnes_pop[mama_index]
            family_fitness.append(m4ms)
            m4mr = list(pop[mama_index])
            family_fitur.append(m4mr)

            p4ps = fitnes_pop[papa_index]
            family_fitness.append(p4ps)
            p4pr = list(pop[papa_index])
            family_fitur.append(p4pr)

            #crossover dan mutasi
            bin_mama = list(pop[mama_index])
            bin_papa = list(pop[papa_index])
            anak_binary = lib.crossover(bin_mama, bin_papa, panjang_fitur,  jumlah_titik = jumlah_titik, prob_mutasi = prob_mutasi)

            #mencari fitness untuk child
            a1 = anak_binary[0]
            a2 = anak_binary[1]
            a3 = anak_binary[2]
    #         print(a2)

            if sum(a1)<=0:
                a1[-1]=1
                a1[-2]=1
            if sum(a2)<=0:
                a2[-1]=1
                a2[-2]=1
            if sum(a3)<=0:
                a3[-1]=1
                a3[-2]=1

            str_a1 = lib.int_to_str(a1)
            if str_a1 in pop_used:
                family_fitur.append(a1)
                family_fitness.append(pop_used[str_a1])

            else:
                family_fitur.append(a1)
                fitness_anak_1 = lib.fitness_kf(data, label, a1, fitur, alpha = alpha, metode = metode, K=K)
                family_fitness.append(fitness_anak_1)
                pop_used.update({str_a1:fitness_anak_1})

            str_a2 = lib.int_to_str(a2)
            if str_a2 in pop_used:
                family_fitur.append(a2)
                family_fitness.append(pop_used[str_a2])
            else:
                family_fitur.append(a2)
                fitness_anak_2 = lib.fitness_kf(data, label, a2, fitur, alpha = alpha, metode = metode, K=K)
                family_fitness.append(fitness_anak_2)
                pop_used.update({str_a2:fitness_anak_2})

            if additional_child == True:
                str_a3 = lib.int_to_str(a3)
                if str_a3 in pop_used:
                    family_fitur.append(a3)
                    family_fitness.append(pop_used[str_a3])
                else:
                    family_fitur.append(a3)
                    fitness_anak_3 = lib.fitness_kf(data, label, a3, fitur, alpha = alpha, metode = metode, K=K)
                    family_fitness.append(fitness_anak_3)
                    pop_used.update({str_a3:fitness_anak_3})


            #mencari fitness terbaik untuk satu keluarga
            best_family_fitness = max(family_fitness)
            bf_index = family_fitness.index(best_family_fitness)
            best_family_fitur = list(family_fitur[bf_index])

            best_fitness_list.append(best_family_fitness)
            best_fitur_list.append(best_family_fitur)

            best_fitness_list_per_generasi.append(best_family_fitness)
            best_fitur_list_per_generasi.append(best_family_fitur)
            best_jumlah_fitur_list.append(sum(best_family_fitur))

    #         print("   ",best_family_fitness)
            print("   ",lib.tampil_finess(best_family_fitness),"|",str(sum(best_family_fitur))+"|"+str(len(best_family_fitur)),"|",gen)

        pop = list(best_fitur_list_per_generasi)
    #     pop.append(features_1_bin)
        fitnes_pop = list(best_fitness_list_per_generasi)
    #     fitnes_pop.append(fns_1)

        if len(pop)<2 or max(best_fitness_list)>=target:
            print("="*70)
            print("Best",max(best_fitness_list))
    #         print("Generasi ke-",best_generasi_list[best_fitness_list.index(max(best_fitness_list))])
            good_fitur = best_fitur_list[best_fitness_list.index(max(best_fitness_list))]
            best_fitness_list.index(max(best_fitness_list))

            best_fitur = np.array(fitur)[lib.get_index(best_fitur_list[-1])]
            best_score = best_fitness_list[-1]

            good_fitur__ = np.array(fitur)[lib.get_index(good_fitur)] 
            print('jumlah fitur     ', sum(good_fitur))
            print('jumlah fitur asli', panjang_fitur)
            print('good_fitur')

            dict_0919 = {
                'best_score':best_score,
                'best_fitur':np.array(best_fitur),
                'good_fitur':np.array(good_fitur__),
                'best_fitur_bin':np.array(best_fitur_list[-1]),
                'good_fitur_bin':np.array(good_fitur),
            }
            return dict_0919