class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:

        self.phoneme_table = phoneme_table
        self.vocabulary = vocabulary
        self.best_state = None

        # Generating a reverse phoneme table
        self.reverse_phoneme = {}
        for key, values in self.phoneme_table.items():
            for value in values:
                if value in self.reverse_phoneme:
                    self.reverse_phoneme[value].append(key)
                else:
                    self.reverse_phoneme[value] = [key]

    def asr_corrector(self, environment):

        num_compute_calls = 0
        self.best_state = environment.init_state

        original_text = environment.init_state
        k = 8

        # Spliting the text into tokens
        n = len(original_text)
        original_tokens = []
        for t in original_text:
            original_tokens.append(t)

        "------------------------------------------------------------------------------------------------------------"
        # STATE FORMULATION
        # curr_samples contains the current best k texts to be considered. The first element of tuple is the tokenized form of current sample ignoring the first or last added words,
        # second element is the bitstring:- 1 represents changes
        # Third element if the first word that might be added
        # 4th is the last word that might be added
        # 5th is the cost of the total text(Including the first or last added words)
        # The last field stores the indices of the tokens where the change has occured

        original_cost = environment.compute_cost(original_text)
        curr_samples = [
            (original_tokens, "0" * n, "", "", original_cost, [])
        ]
        num_compute_calls += 1

        new_samples = []

        "------------------------------------------------------------------------------------------------------------"
        # all possible cases by adding first word
        for w1 in self.vocabulary:
            new_samples.append(
                (
                    original_tokens,
                    "0" * n,
                    w1,
                    "",
                    environment.compute_cost(w1 + " " + original_text),
                    []
                )
            )

            num_compute_calls += 1

        new_samples = sorted(new_samples, key=lambda x: x[4])

        if len(new_samples) > 3:
            new_samples = new_samples[:3]

        curr_samples = curr_samples + new_samples

        new_samples = []
        "------------------------------------------------------------------------------------------------------------"
        # all possible cases by adding second word over the best solutions after first word addition
        for sample in curr_samples:
            newtext = "".join(sample[0])
            if sample[2] != "":
                newtext = sample[2] + " " + newtext

            for w2 in self.vocabulary:
                new_samples.append(
                    (
                        original_tokens,
                        "0" * n,
                        sample[2],
                        w2,
                        environment.compute_cost(newtext + " " + w2),
                        []
                    )
                )
                num_compute_calls += 1

        curr_samples = curr_samples + new_samples
        curr_samples = sorted(curr_samples, key=lambda x: x[4])

        if len(curr_samples) > 10:
            curr_samples = curr_samples[:10]

        "------------------------------------------------------------------------------------------------------------"
        # all character changes from the reverse phoneme table
        for i in range(n):
            new_samples = []

            for sample in curr_samples:

                if sample[1][i] == "0" and original_tokens[i] != " ":
                    #  for single letter swapped with single/double letter
                    if original_tokens[i] in self.reverse_phoneme:
                        for new_letter in self.reverse_phoneme[original_tokens[i]]:
                            new_tokens = sample[0].copy()
                            new_tokens[i] = new_letter
                            new_bitstring = sample[1][:i] + "1" + sample[1][i + 1 :]
                            new_text = "".join(new_tokens)
                            new_changes_arr = sample[5].copy()
                            if(len(new_changes_arr) == 0):
                                new_changes_arr = [i]
                            else:
                                new_changes_arr.append(i)
                            if sample[2] != "":
                                new_text = sample[2] + " " + new_text

                            if sample[3] != "":
                                new_text += " " + sample[3]

                            new_samples.append(
                                (
                                    new_tokens,
                                    new_bitstring,
                                    sample[2],
                                    sample[3],
                                    environment.compute_cost(new_text),
                                    new_changes_arr
                                )
                            )
                            num_compute_calls += 1

                    # for 2 letters swapped with a single/double letter
                    if (
                        i < n - 1
                        and original_tokens[i + 1] != " "
                        and (
                            (original_tokens[i] + original_tokens[i + 1])
                            in self.reverse_phoneme
                        )
                    ):

                        for new_letter in self.reverse_phoneme[
                            original_tokens[i] + original_tokens[i + 1]
                        ]:
                            new_tokens = sample[0].copy()
                            new_tokens[i] = new_letter
                            new_tokens[i + 1] = ""
                            new_bitstring = sample[1][:i] + "11" + sample[1][i + 2 :]
                            new_text = "".join(new_tokens)
                            new_changes_arr = sample[5].copy()
                            if(len(new_changes_arr) == 0):
                                new_changes_arr = [i]
                            else:
                                new_changes_arr.append(i)

                            if sample[2] != "":
                                new_text = sample[2] + " " + new_text

                            if sample[3] != "":
                                new_text += " " + sample[3]

                            new_samples.append(
                                (
                                    new_tokens,
                                    new_bitstring,
                                    sample[2],
                                    sample[3],
                                    environment.compute_cost(new_text),
                                    new_changes_arr
                                )
                            )
                            num_compute_calls += 1

            curr_samples = curr_samples + new_samples
            curr_samples = sorted(curr_samples, key=lambda x: x[4])

            if len(curr_samples) > k:
                curr_samples = curr_samples[:k]

            best_text = "".join(curr_samples[0][0])

            if curr_samples[0][2] != "":
                best_text = curr_samples[0][2] + " " + best_text
            if curr_samples[0][3] != "":
                best_text += " " + curr_samples[0][3]

            self.best_state = best_text

        if len(curr_samples) > k:
            curr_samples = curr_samples[:k]
        
        new_samples = []

        "------------------------------------------------------------------------------------------------------------"
        # reverting back the first/last word changes to check for better solutions
        for sample in curr_samples:
            if (sample[2] == "" and sample[3] != "") or (
                sample[2] != "" and sample[3] == ""
            ):
                new_tokens = sample[0].copy()
                new_text = "".join(new_tokens)
                new_samples.append(
                    (new_tokens, sample[1], "", "", environment.compute_cost(new_text), sample[5])
                )

                num_compute_calls += 1

            elif sample[2] != "" and sample[3] != "":
                new_tokens = sample[0].copy()
                new_text1 = "".join(new_tokens)
                new_text2 = sample[2] + " " + new_text1
                new_text3 = new_text1 + " " + sample[3]

                new_samples.append(
                    (new_tokens, sample[1], "", "", environment.compute_cost(new_text1), sample[5])
                )
                new_samples.append(
                    (
                        new_tokens,
                        sample[1],
                        sample[2],
                        "",
                        environment.compute_cost(new_text2),
                        sample[5]
                    )
                )
                new_samples.append(
                    (
                        new_tokens,
                        sample[1],
                        "",
                        sample[3],
                        environment.compute_cost(new_text3),
                        sample[5]
                    )
                )

                num_compute_calls += 3

        curr_samples = curr_samples + new_samples
        curr_samples = sorted(curr_samples, key=lambda x: x[4])

        best_text = "".join(curr_samples[0][0])

        if curr_samples[0][2] != "":
            best_text = curr_samples[0][2] + " " + best_text
        if curr_samples[0][3] != "":
            best_text += " " + curr_samples[0][3]

        self.best_state = best_text


        "------------------------------------------------------------------------------------------------------------"
        # reverting back the changes made in the tokens(single/double letters) to check for better solutions
        k = 4
        new_curr_samples = []
        for sample in curr_samples[:4]:
            new_samples = []
            change_array = sample[5].copy()
            new_samples.append(sample)
            for idx in change_array:
                new_new_samples = []
                for sample_new in new_samples:
                    new_tokens = sample_new[0].copy()
                    new_bitstr = sample_new[1]
                    new_changes_arr = sample_new[5].copy()
                    new_tokens[idx] = original_tokens[idx]
                    new_bitstr = new_bitstr[:idx] + "0" +  new_bitstr[idx+1:]
                    if idx in new_changes_arr:
                        new_changes_arr.remove(idx)
                    if(idx<n-1 and new_tokens[idx+1]==""):
                        new_tokens[idx+1] = original_tokens[idx+1]
                        new_bitstr = new_bitstr[:idx+1] + "0" +  new_bitstr[idx+2:]
                    new_text = "".join(new_tokens)
                    if(sample_new[2] != ""):
                        new_text = sample_new[2] + " " + new_text
                    if(sample_new[3] != ""):
                        new_text = new_text + " " + sample_new[3]
                    new_new_samples.append(
                        (new_tokens,
                        new_bitstr,
                        sample_new[2],
                        sample_new[3],
                        environment.compute_cost(new_text),
                        new_changes_arr
                        )
                    )
                    num_compute_calls += 1
                
                new_samples = new_new_samples + new_samples
                new_samples = sorted(new_samples, key=lambda x:x[4])

                if len(new_samples) > k:
                    new_samples =  new_samples[:k]

            new_curr_samples = new_curr_samples + new_samples
            new_curr_samples = sorted(new_curr_samples, key=lambda x:x[4])
            if(len(new_curr_samples) > k):
                new_curr_samples = new_curr_samples[:k]
            
        curr_samples = curr_samples + new_curr_samples
        curr_samples = sorted(curr_samples, key=lambda x:x[4])
        if(len(curr_samples) > k):
            curr_samples = curr_samples[:k]



        best_text = "".join(curr_samples[0][0])
        if curr_samples[0][2] != "":
            best_text = curr_samples[0][2] + " " + best_text
        if curr_samples[0][3] != "":
            best_text += " " + curr_samples[0][3]

        self.best_state = best_text


        curr_best_sample = curr_samples[0]
        curr_best_tokens = curr_best_sample[0].copy()
        curr_best_text = "".join(curr_best_tokens)

        new_samples = []

        for w1 in self.vocabulary:
            new_samples.append(
                (
                    curr_best_tokens,
                    curr_best_sample[1],
                    w1,
                    "",
                    environment.compute_cost(w1 + " " + curr_best_text),
                    []
                )
            )
            num_compute_calls += 1
        
        new_samples = sorted(new_samples, key=lambda x : x[4])
        curr_best_sample_2 = new_samples[0]
        curr_best_tokens_2 = curr_best_sample_2[0].copy()
        curr_best_text_2 = "".join(curr_best_tokens_2)
        
        best_text = curr_best_text_2
        if curr_best_sample_2[2] != "":
            best_text = curr_best_sample_2[2] + " " + best_text
        if curr_best_sample_2[3] != "":
            best_text += " " + curr_best_sample_2[3]
        self.best_state = best_text

        new_samples = []

        for w1 in self.vocabulary:
            new_samples.append(
                (
                    curr_best_tokens,
                    curr_best_sample[1],
                    "",
                    w1,
                    environment.compute_cost(curr_best_text + " " + w1),
                    []
                )
            )

            new_samples.append(
                (
                    curr_best_tokens_2,
                    curr_best_sample_2[1],
                    curr_best_sample_2[2],
                    w1,
                    environment.compute_cost(curr_best_sample_2[2] + " " + curr_best_text_2 + " " + w1),
                    []
                )
            )

            num_compute_calls += 2

        curr_samples.append(curr_best_sample_2)
        curr_samples = curr_samples + new_samples
        curr_samples = sorted(curr_samples, key=lambda x : x[4] )


        "------------------------------------------------------------------------------------------------------------"
        # final best solution
        best_text = "".join(curr_samples[0][0])
        if curr_samples[0][2] != "":
            best_text = curr_samples[0][2] + " " + best_text
        if curr_samples[0][3] != "":
            best_text += " " + curr_samples[0][3]
        self.best_state = best_text

        "------------------------------------------------------------------------------------------------------------"
        # debug statements
        """
        with open("debug_outputs.txt", "a") as file:
            file.write("start sentence was:- \n" + environment.init_state + "\n")
            file.write("Cost of start sentence is:- " + str(environment.compute_cost(environment.init_state)) + "\n")
            file.write("corrected sentence is:- \n" + self.best_state + "\n")
            file.write("Cost of corrected sentence is:- " + str(environment.compute_cost(self.best_state)) + "\n")
            file.write("Time taken in minutes is : " + str(t) + "\n")
            file.write("Number of cost compute calls: " + str(num_compute_calls) + "\n\n\n")

        print("start sentence was:- \n" + environment.init_state + "\n")
        print("Cost of start sentence is:- " + str(environment.compute_cost(environment.init_state)) + "\n")
        print("corrected sentence is:- \n" + self.best_state + "\n")
        print("Cost of corrected sentence is:- " + str(environment.compute_cost(self.best_state)) + "\n")
        print("Time taken in minutes is : " + str(t) + "\n")
        print("Number of cost compute calls: " + str(num_compute_calls) + "\n\n\n")
        """

""" ----------- SIMULATED ANNEALING: ----------------------------------------------------------------------------------------------------
    def simulated_annealin(self, environment, t=100000, cooling_rate=0.25, num_iterations=1000):
        curr_state = environment.init_state
        temp = t
        next_state = ""
        next_energy = 0
        delta_energy = 0
        curr_energy = environment.compute_cost(curr_state)
        print(f"entered SA function, with curr_state = {curr_state} and curr_energy={curr_energy}")
        for i in range(num_iterations):
            print(f"iteration num: {i}")
            temp = temp - cooling_rate*temp
            # next_state = "HELLLLOOOOO"
            next_state = self.get_random_neighbor(curr_state)
            next_energy = environment.compute_cost(next_state)
            delta_energy = curr_energy - next_energy
            print(f'iteration: {i} , curr_string= {curr_state}, next_string= {next_state}, delta_energy = {delta_energy}')
            if(delta_energy > 0):
                # if the energy is decreasing then we are moving towards more stable state
                curr_state = next_state
                curr_energy = next_energy
            else:
                # p = e^(delta_energy / temp)
                p = math.exp(delta_energy / temp)
                rand_value  = round(random.random(), 2)
                if(p > rand_value):
                    curr_state = next_state
                    curr_energy = next_energy
            self.best_state = curr_state
"""

""" ------------- HILL-CLIMB ---------------------------------------------------------------------------------------------------------
    def asr_corrector(self, environment):
        t1 = time.time()
        text = environment.init_state
        tokens = []
        for c in text:
            tokens.append(c)
        n = len(tokens)
        bitstr = "0"*n
        oldchar = ''
        oldcost = environment.compute_cost(text)
        initial_cost = oldcost
        for i in range(0, n):
            # print(f"iteration: {i} with token: {tokens[i]}")
            if(tokens[i] != ' '):
                if tokens[i] in self.reverse_phoneme.keys():
                    oldchar = tokens[i]
                    for k in self.reverse_phoneme[tokens[i]]:
                        tokens[i] = k
                        newtext = "".join(tokens)
                        newcost = environment.compute_cost(newtext)
                        if(newcost < oldcost):
                            text = newtext
                            oldcost = newcost
                        else:
                            tokens[i] = oldchar
        self.best_state = text
        t2 = time.time()
        with open("hc_results.txt", "a") as file:
            file.write(
                "\n\n"
                + f"{environment.init_state} \n"
                + f"{initial_cost} \n"
                + f"{text} \n"
                + f"{oldcost} \n"
                + f"elapsed time: {(t2-t1)/60} minutes \n"
            )
"""

""" ------------------- RANDOM-WALK -------------------------------------------------------------------------------------------------
    def random_walk(self, environment):
        self.best_state = environment.init_state
        text = environment.init_state
        tokens = []
        for c in text:
            tokens.append(c)
        n = len(text)
        bitstr = "0"*n

        m = len(tokens)
        num_active = 0
        oldtext = "".join(tokens)
        newtext = ""
        oldcost = environment.compute_cost(oldtext)
        active_token_nums = []
        for i in range(0, n):
            active_token_nums.append(i)
        t1 = time.time()
        st_time = t1
        ed_time = t1
        i = 0
        while(i<10):
            i = i + 1
            ed_time = time.time()
            rand_idx = random.choice(active_token_nums)
            active_token_nums.remove(rand_idx)
            if(bitstr[rand_idx] != '1'):
                if(tokens[rand_idx] != ' '):
                    oldchar = tokens[rand_idx]
                    for newchar in self.phoneme_table[oldchar]:
                        tokens[rand_idx] = newchar
                        newtext = "".join(tokens)
                        newcost = environment.compute_cost(newtext)
                        if(newcost > oldcost):
                            tokens[rand_idx] = oldchar
                        else:
                            oldcost = newcost
                            self.best_state = newtext
            if(num_active == m):
                break
        print("elapsed time: ", ed_time-st_time)
        print("changed text: " , newtext)
"""
