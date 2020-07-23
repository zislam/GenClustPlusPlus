/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package weka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.rules.DecisionTableHashKey;
import static weka.clusterers.SimpleKMeans.CANOPY;
import static weka.clusterers.SimpleKMeans.FARTHEST_FIRST;
import static weka.clusterers.SimpleKMeans.KMEANS_PLUS_PLUS;
import static weka.clusterers.SimpleKMeans.RANDOM;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.core.converters.ConverterUtils.DataSource;
/**
 * <!-- globalinfo-start -->
 * Class implementing GenClust++ clustering algorithm.<br>
 * For more information, see:<br>
 * <br>
 * Islam, M. Z., Estivill-Castro, V., Rahman, M. A. and Bossomaier, T. (2018).
 * Combining K-Means and a Genetic Algorithm through a Novel Arrangement of
 * Genetic Operators for High Quality Clustering. Expert Systems with
 * Applications.
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{adnan2017forest,
 *   title={Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering},
 *   author={Islam, M. Z., Estivill-Castro, V., Rahman, M. A. and Bossomaier, T.},
 *   journal={Expert Systems with Applications},
 *   year={2018},
 *   volume={91},
 *   pages={402-417},
 *   publisher={Elsevier}
 * }
 * </pre>
 *
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * Valid options are:
 *
 * <pre> -G &lt;num generations&gt;
 *  Number of generations for genetic algorithm.
 *  (default 60)</pre>
 *
 *
 * <pre> -P &lt;popn size&gt;
 *  Initial population size for generic algorithm.
 *  (default 30)</pre>
 *
 * <pre> -N &lt;max num iterations&gt;
 *  Max iterations for initial k-means.
 *  (default 60)</pre>
 *
 * <pre> -Q &lt;max num iterations&gt;
 *  Max iterations for quick k-means.
 *  (default 15)</pre>
 *
 * <pre> -F &lt;max num iterations&gt;
 *  Max iterations for final run of k-means.
 *  (default 50)</pre>
 *
 * <pre> -D &lt;duplicate threshold&gt;
 *  Threshold for difference between two genes for them to be considered
 *  duplicates. Always between 0 and 1.
 *  (default 0)</pre>
 * 
 * <pre> -M 
 *  Do not replace missing values with a global mean / mode.
 *  (default false)</pre>
 *
 * <!-- options-end -->
 *
 * @author Michael Furner (mfurner at csu dot edu dot au)
 * @version 1.0
 */
public class GenClustPlusPlus extends RandomizableClusterer
        implements TechnicalInformationHandler {

    private static final long serialVersionUID = -7247404718496233612L;

    private Instances m_data;
    private int m_numberOfClusters = 0;
    private int m_numberOfGenerations = 60;
    private ManhattanDistance m_distFunc;
    private int m_initialPopulationSize = 30;
    private int m_maxKMeansIterationsInitial = 60;
    private int m_maxKMeansIterationsQuick = 15;
    private int m_maxKMeansIterationsFinal = 50;
    private double m_duplicateThreshold = 0;
    private int m_startChromosomeSelectionGeneration = 50;
    private MKMeans m_bestChromosome;
    private double m_bestFitness;
    private int m_bestFitnessIndex;
    private MKMeans m_builtClusterer = null;
    private Random m_rand = null;
    private ReplaceMissingValues m_ReplaceMissingFilter = null;
    private boolean m_dontReplaceMissing = false;

    public static final int SUPPLIED = 4;
    public static final Tag[] TAGS_SELECTION_MK = {new Tag(RANDOM, "Random"),
        new Tag(KMEANS_PLUS_PLUS, "k-means++"), new Tag(CANOPY, "Canopy"),
        new Tag(FARTHEST_FIRST, "Farthest first"), new Tag(SUPPLIED, "Supplied Centroids")
    };

    /**
     * Returns the number of clusters.
     *
     * @return the number of clusters generated for a training dataset.
     * @throws Exception if number of clusters could not be returned
     * successfully
     */
    @Override
    public int numberOfClusters() {
        return m_numberOfClusters;
    }

    /**
     * the default constructor.
     */
    public GenClustPlusPlus() {
        super();

        m_SeedDefault = 10;
        setSeed(m_SeedDefault);
    }

    /**
     * Generates a clusterer. Has to initialize all fields of the clusterer that
     * are not being set via options.
     *
     * @param data set of instances serving as training data
     * @throws Exception if the clusterer has not been generated successfully
     */
    @Override
    public void buildClusterer(Instances data) throws Exception {

        m_rand = new Random(getSeed());
        getCapabilities().testWithFail(data);

        m_data = new Instances(data);

        //check if every value is missing - if it is, just return the default
        //result of a SimpleKMeans
        boolean allMissing = true;
        for (int j = 0; j < m_data.numAttributes(); j++) {
            if (m_data.attributeStats(j).missingCount != m_data.numInstances()) {
                allMissing = false;
            }
        }
        if (allMissing) {
            MKMeans sk = new MKMeans();
            sk.buildClusterer(m_data);
            m_builtClusterer = sk;
            m_numberOfClusters = sk.numberOfClusters();
            return;
        }

        m_ReplaceMissingFilter = new ReplaceMissingValues();
        m_data.setClassIndex(-1);
        if (!m_dontReplaceMissing) {
            m_ReplaceMissingFilter.setInputFormat(m_data);
            m_data = Filter.useFilter(m_data, m_ReplaceMissingFilter);
        }

        //Get the initial population for the genetic algorithm
        //prepare the distance function early for ease of computation with vicus similarity
        m_distFunc = new ManhattanDistance(m_data);
        MKMeans[] initialPopulation = generateInitialPopulation(m_data);
        if(initialPopulation == null) {
            return;
        }

        //select best chromosome
        m_bestFitness = Double.NEGATIVE_INFINITY;
        m_bestFitnessIndex = Integer.MIN_VALUE;
        for (int i = 0; i < initialPopulation.length; i++) {
            double thisFitness = fitness(initialPopulation[i]);
            if (thisFitness > m_bestFitness) {
                if (thisFitness == Double.POSITIVE_INFINITY) {
                    m_builtClusterer = initialPopulation[i];
                    m_numberOfClusters = initialPopulation[i].getClusterCentroids().size();
                    return;
                }
                m_bestFitness = thisFitness;
                m_bestFitnessIndex = i;
            }
        }

        m_bestChromosome = new MKMeans(initialPopulation[m_bestFitnessIndex]);

        //perform probabalistic selection
        MKMeans[] selectedPopulation = probabilisticSelection(initialPopulation);
        MKMeans[] previous = new MKMeans[selectedPopulation.length];
        //Loop over user defined iterations of genetic algorithm
        for (int g = 0; g <= m_numberOfGenerations; g++) {
            MKMeans[] crossoverPopulation;
            if (((g + 1) % 10) != 0) {
                //crossover
                crossoverPopulation = crossover(selectedPopulation);
                
            } else {
                //every 10 iterations perform probabalistic cloning and elitism
                crossoverPopulation = probabilisticCloning(selectedPopulation);

                crossoverPopulation = elitism(crossoverPopulation);

                //perform quick mkmeans
                for (int i = 0; i < crossoverPopulation.length; i++) {

                    MKMeans mk;
                    do {
                        mk = new MKMeans();
                        mk.setSeed(m_rand.nextInt());
                        mk.setInitializationMethod(new SelectedTag(GenClustPlusPlus.SUPPLIED, GenClustPlusPlus.TAGS_SELECTION_MK));
                        mk.setInitial(crossoverPopulation[i].getClusterCentroids());
                        mk.setMaxIterations(m_maxKMeansIterationsQuick);
                        mk.setDontReplaceMissingValues(m_dontReplaceMissing);
                        mk.setPreserveInstancesOrder(true);
                        mk.buildClusterer(m_data, m_distFunc);
                        
                        if (mk.getClusterCentroids().numInstances() <= 1) {
                            //crossoverPopulation[i] = mutate(crossoverPopulation[i].getClusterCentroids());
                            
                            //Produce random k value between 2 and sqrt(|data|)
                            int randomK = m_rand.nextInt((int) (Math.sqrt(data.size()) - 2)) + 2;
                            //execute k means
                            MKMeans newChromosome = new MKMeans();
                            newChromosome.setSeed(m_rand.nextInt());
                            newChromosome.setNumClusters(randomK);
                            newChromosome.setDontReplaceMissingValues(m_dontReplaceMissing);
                            newChromosome.setPreserveInstancesOrder(true);
                            newChromosome.buildClusterer(data, m_distFunc);
                            crossoverPopulation[i] = new MKMeans(newChromosome);
                                    
                        }
                        
                    } while (mk.getClusterCentroids().numInstances() <= 1);
                    crossoverPopulation[i] = mk;

                }
            }

            //Perform elitism
            crossoverPopulation = elitism(crossoverPopulation);
                    
            //Perform mutation
            crossoverPopulation = mutation(crossoverPopulation);

            //Perform elitism
            crossoverPopulation = elitism(crossoverPopulation);

            if (m_builtClusterer != null) {
                return;
            }

            if (g > m_startChromosomeSelectionGeneration) {
                //Chromosome selection                
                FitnessContainer[] current = new FitnessContainer[crossoverPopulation.length];
                FitnessContainer[] prev = new FitnessContainer[previous.length];
                for (int i = 0; i < crossoverPopulation.length; i++) {
                    current[i] = new FitnessContainer(fitness(crossoverPopulation[i]), crossoverPopulation[i]);
                }
                for (int i = 0; i < previous.length; i++) {
                    prev[i] = new FitnessContainer(fitness(previous[i]), previous[i]);
                }
                
                FitnessContainer[] sorted = new FitnessContainer[crossoverPopulation.length * 2];
                for (int i = 0; i < crossoverPopulation.length; i++) {
                    sorted[i] = current[i];
                }
                int t = 0;
                for (int i = crossoverPopulation.length; i < crossoverPopulation.length*2; i++) {
                    sorted[i] = prev[t++];
                }
                
                Arrays.sort(sorted, Collections.reverseOrder());
                for (int i = 0; i < crossoverPopulation.length; i++) {
                    selectedPopulation[i] = new MKMeans(sorted[i].clustering);
                }

            } else {
                //for the first 10 iterations Pm is the new population with 
                //no chromosome selection
                for (int i = 0; i < crossoverPopulation.length; i++) {
                    selectedPopulation[i] = new MKMeans(crossoverPopulation[i]);
                }

            }
            for (int i = 0; i < selectedPopulation.length; i++) {
                previous[i] = new MKMeans(selectedPopulation[i]);
            }
//            for (int i = 0; i < selectedPopulation.length; i++) {
//               double f = fitness(selectedPopulation[i]);
//               System.out.println(f);
//            }
//            System.out.println("Fitness max: " + m_bestFitness);

        } //end of main loop

        double newBestFitness = Double.MIN_VALUE;
        int newBestIndex = Integer.MAX_VALUE;
        for (int i = 0; i < selectedPopulation.length; i++) {
            double f = fitness(selectedPopulation[i]);

            if (f > newBestFitness) {
                newBestFitness = f;
                newBestIndex = i;
            }
        }

//        if (newBestFitness == 0) {
//            m_builtClusterer = selectedPopulation[newBestIndex];
//        }
        m_bestChromosome = new MKMeans(selectedPopulation[newBestIndex]);
        m_bestFitness = newBestFitness;

        MKMeans finalRun = new MKMeans();
        finalRun.setSeed(m_rand.nextInt());
        finalRun.setInitializationMethod(new SelectedTag(GenClustPlusPlus.SUPPLIED, GenClustPlusPlus.TAGS_SELECTION_MK));
        finalRun.setInitial(m_bestChromosome.getClusterCentroids());
        finalRun.setDontReplaceMissingValues(m_dontReplaceMissing);
        finalRun.setPreserveInstancesOrder(true);
        finalRun.setMaxIterations(m_maxKMeansIterationsFinal);
        finalRun.buildClusterer(m_data, m_distFunc);
        m_builtClusterer = finalRun;
        m_numberOfClusters = m_builtClusterer.getClusterCentroids().size();

    }

    /**
     * Creates the initial chromosomes population using the data
     *
     * @param data
     * @return a set of chromosomes for the initial population
     * @throws Exception
     */
    private MKMeans[] generateInitialPopulation(Instances data) throws Exception {

        int maxK = 3 * m_initialPopulationSize / 10 + 1;
        int numberOfChromosomes = 5 * (maxK - 1) * 2;

        MKMeans[] population = new MKMeans[numberOfChromosomes];
        int chromosomeCount = 0;

        for (int i = 2; i <= maxK; i++) {
            for (int j = 0; j < 5; j++) {
                //execute k means
                MKMeans chromosome = new MKMeans();
                int failCount = -1;
                do {
                    failCount++;
                    if(failCount > 100) {
                        System.out.println("Unable to cluster this dataset using genetic algorithm, try imputing missing values first.");
                        chromosome.setUnable(true);
                        m_builtClusterer = chromosome;
                        m_numberOfClusters = m_builtClusterer.getClusterCentroids().size();
                        return null;
                    }
                    chromosome.setSeed(m_rand.nextInt());
                    chromosome.setNumClusters(i);
                    chromosome.setPreserveInstancesOrder(true);
                    chromosome.setMaxIterations(m_maxKMeansIterationsInitial);
                    chromosome.setDontReplaceMissingValues(m_dontReplaceMissing);
                    chromosome.buildClusterer(data, m_distFunc);
                } while (chromosome.getNumClusters() < 2);
                population[chromosomeCount++] = chromosome;

            }
        }
        for (int i = 0; i < maxK - 1; i++) {

            //Produce random k value between 2 and sqrt(|data|)
            int randomK = m_rand.nextInt((int) (Math.sqrt(data.size()) - 2)) + 2;

            for (int j = 0; j < 5; j++) {
                //execute k means
                MKMeans chromosome = new MKMeans();
                chromosome.setSeed(m_rand.nextInt());
                chromosome.setNumClusters(randomK);
                chromosome.setDontReplaceMissingValues(m_dontReplaceMissing);
                chromosome.setPreserveInstancesOrder(true);
                do {
                    chromosome.buildClusterer(data, m_distFunc);
                } while (chromosome.getNumClusters() < 2);
                
                population[chromosomeCount++] = chromosome;

            }
        }

        return population;

    }

    /**
     * Measures the fitness of a chromosome using DB Index
     *
     * @param chromosome
     * @return
     * @throws Exception
     */
    private double fitness(MKMeans chromosome) {

        //calculate fitness using davies bouldin index
        EuclideanDistance eu = new EuclideanDistance(m_data);
        Instances centroids = chromosome.getClusterCentroids();

        double[] Si = new double[centroids.numInstances()];
        int[] Ti = new int[centroids.numInstances()];
        int[] clustSize = new int[centroids.numInstances()];
        int numClust = centroids.numInstances();
        if(numClust == 1) {
            return 0.0;
        }

        int[] assignments = null;
        try {
            assignments = chromosome.getAssignments();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        for (int j = 0; j < assignments.length; j++) {

            //int thisAssignment = assignments[j];
            try {
                int thisAssignment = chromosome.clusterInstance(m_data.get(j));   
                clustSize[thisAssignment]++;
                Si[thisAssignment] += Math.abs(eu.distance(m_data.get(j), centroids.get(thisAssignment)));
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }

        }

        for (int i = 0; i < centroids.numInstances(); i++) {
            Ti[i] = clustSize[i];
            if (clustSize[i] != 0) {
                Si[i] = Si[i] / Ti[i];
            } else {
                //nonEmptyClusters--;
                return 0.0;
                //Si[i] = 0;
            }
        }

        double DB = 0;
        for (int i = 0; i < centroids.numInstances(); i++) {

            if (clustSize[i] == 0) {
                continue;
            }

            double max = Double.NEGATIVE_INFINITY;
            int maxIndex = Integer.MIN_VALUE;

            for (int j = 0; j < centroids.numInstances(); j++) {

                if (i == j) {
                    continue;
                }

                double Rij = Si[i] + Si[j] / Math.abs(eu.distance(centroids.get(i), centroids.get(j)));
                if (Rij > max) {
                    max = Rij;
                    maxIndex = j;
                }

            }

            if (max != Double.NEGATIVE_INFINITY) {
                DB += max;
            }

        }

        DB /= numClust;
        return 1.0 / DB;

    }

    /**
     *
     * @param population
     * @return
     * @throws Exception
     */
    private MKMeans[] probabilisticSelection(MKMeans[] population) {

        MKMeans[] selectedPopulation = new MKMeans[m_initialPopulationSize];
        int currentSelections = 0;

        double[] fitnessArray = new double[population.length];
        boolean[] usedChromosome = new boolean[population.length];
        double[] TkArray = new double[population.length / 5];
        int[] kArray = new int[population.length / 5];
        double sumTk = 0;

        //get fitness values over k values
        for (int i = 0; i < population.length; i += 5) {

            kArray[i / 5] = population[i].getNumClusters();
            double Tk = 0;

            for (int j = 0; j < 5; j++) {

                double fit = fitness(population[i + j]);
                fitnessArray[i + j] = fit;
                Tk += fit;

            }
            sumTk += Tk;
            TkArray[i / 5] = Tk;

        }

        //select based on probabilities
        while (currentSelections < selectedPopulation.length) {

            double p = m_rand.nextDouble();
            double cumulativeProbability = 0.0;
            for (int i = 0; i < TkArray.length; i++) {
                cumulativeProbability += TkArray[i] / sumTk;
                if (p <= cumulativeProbability) { //we've selected it

                    double max = Double.NEGATIVE_INFINITY;
                    int maxIndex = Integer.MAX_VALUE;
                    for (int j = 0; j < 5; j++) {
                        if (fitnessArray[i + j] > max && !usedChromosome[i + j]) {
                            max = fitnessArray[i + j];
                            maxIndex = i + j;
                        }
                    }

                    MKMeans selected;
                    if (maxIndex == Integer.MAX_VALUE) {
                        try {
                            MKMeans chromosome = new MKMeans();
                            chromosome.setSeed(m_rand.nextInt());
                            chromosome.setNumClusters(kArray[i]);
                            chromosome.setPreserveInstancesOrder(true);
                            chromosome.setDontReplaceMissingValues(m_dontReplaceMissing);
                            chromosome.buildClusterer(m_data, m_distFunc);
                            selected = chromosome;
                        } catch (Exception ex) {
                            ex.printStackTrace();
                            return null;
                        }
                    } else {
                        usedChromosome[maxIndex] = true;
                        selected = population[maxIndex];
                    }
                    selectedPopulation[currentSelections++] = selected;

                    break;

                }
            }

        }

        return selectedPopulation;

    }

    /**
     * Returns technical information
     *
     * @return technical information
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.TITLE, "Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering");
        result.setValue(TechnicalInformation.Field.AUTHOR, "Islam, M. Z., Estivill-Castro, V., Rahman, M. A. and Bossomaier, T.");
        result.setValue(TechnicalInformation.Field.YEAR, "2018");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Expert Systems with Applications");
        result.setValue(TechnicalInformation.Field.VOLUME, "91");
        result.setValue(TechnicalInformation.Field.PAGES, "402-417");

        return result;
    }

    private MKMeans[] crossover(MKMeans[] selectedPopulation) {

        Instances[] offspring = new Instances[selectedPopulation.length];
        int offspringCounter = 0;

        FitnessContainer[] sorted = new FitnessContainer[selectedPopulation.length];
        double fitnessSum = 0;

        //get the fitnesses and sort the k means solutions
        for (int i = 0; i < sorted.length; i++) {
            double thisF = fitness(selectedPopulation[i]);
            fitnessSum += thisF;
            sorted[i] = new FitnessContainer(thisF, selectedPopulation[i]);
        }
        Arrays.sort(sorted, Collections.reverseOrder());
        
        while (sorted.length > 0) {
            //pick two parents
            MKMeans parentOne = sorted[0].clustering;
            fitnessSum -= sorted[0].fitness;
            sorted[0] = null;
            MKMeans parentTwo = sorted[sorted.length - 1].clustering;

            double p = m_rand.nextDouble();
            double cumulativeProbability = 0;
            for (int i = 1; i < sorted.length; i++) {
                cumulativeProbability += sorted[i].fitness / fitnessSum;
                if (p <= cumulativeProbability) { //we've selected it
                    parentTwo = sorted[i].clustering;
                    fitnessSum -= sorted[i].fitness;
                    sorted[i] = null;
                    break;
                }
            }

            //rearrange operation
            Instances reference = parentOne.getClusterCentroids();
            Instances parentTwoCentroids = parentTwo.getClusterCentroids();

            Instances target = new Instances(reference, 0);
            for (int i = 0; i < reference.numInstances() && parentTwoCentroids.numInstances() > 0; i++) {

                int closestIndex = Integer.MAX_VALUE;
                double distance = Double.MAX_VALUE;

                for (int j = 0; j < parentTwoCentroids.numInstances(); j++) {
                    double d = Math.abs(m_distFunc.distance(reference.get(i), parentTwoCentroids.get(j)));
                    if (d == Double.POSITIVE_INFINITY) {
                        m_distFunc.distance(reference.get(i), parentTwoCentroids.get(j));
                    }
                    if (d < distance) {
                        closestIndex = j;
                        distance = d;
                    }
                }

                if (closestIndex != Integer.MAX_VALUE) {
                    target.add(parentTwoCentroids.get(closestIndex));
                    parentTwoCentroids.remove(closestIndex);
                }

            }

            if (parentTwoCentroids.numInstances() > 0) {
                for (int i = 0; i < parentTwoCentroids.numInstances(); i++) {

                    int closestIndex = Integer.MAX_VALUE;
                    double distance = Double.MAX_VALUE;

                    for (int j = 0; j < target.numInstances(); j++) {
                        double d = Math.abs(m_distFunc.distance(parentTwoCentroids.get(i), target.get(j)));
                        if (d < distance) {
                            distance = d;
                            closestIndex = j;
                        }
                    }

                    target.add(closestIndex + 1, parentTwoCentroids.get(i));

                }
            }

            //conventional crossover operation
            int refRandom =  (reference.numInstances() > 1) 
                    ? m_rand.nextInt(reference.numInstances() - 1) + 1 
                    : 0;
            Instances refFirstHalf = new Instances(reference, 0);
            Instances refSecondHalf = new Instances(reference, 0);
            for (int i = 0; i < reference.numInstances(); i++) {
                if (i < refRandom) {
                    refFirstHalf.add(reference.get(i));
                } else {
                    refSecondHalf.add(reference.get(i));
                }
            }

            int targetRandom =  (target.numInstances() > 1) 
                    ? m_rand.nextInt(target.numInstances() - 1) + 1 
                    : 0;
            Instances targetFirstHalf = new Instances(target, 0);
            Instances targetSecondHalf = new Instances(target, 0);
            for (int i = 0; i < target.numInstances(); i++) {
                if (i < targetRandom) {
                    targetFirstHalf.add(target.get(i));
                } else {
                    targetSecondHalf.add(target.get(i));
                }
            }

            refFirstHalf.addAll(targetSecondHalf);
            targetFirstHalf.addAll(refSecondHalf);
            offspring[offspringCounter++] = refFirstHalf;
            offspring[offspringCounter++] = targetFirstHalf;

            //restart with new sorted array            
            FitnessContainer[] tmp = new FitnessContainer[sorted.length - 2];
            boolean hitSkip = false;
            for (int i = 0; i < tmp.length; i++) {
                int ind = i + 1;
                if (hitSkip || sorted[ind] == null) {
                    ind = i + 2;
                    hitSkip = true;
                }

                tmp[i] = sorted[ind];
            }
            sorted = tmp;

        }

        MKMeans[] offspringMK = new MKMeans[offspring.length];

        //duplicate removal
        for (int i = 0; i < offspring.length; i++) {

            ArrayList<Integer> toRemove = new ArrayList<>();

            for (int j = 0; j < offspring[i].numInstances(); j++) {

                for (int k = j + 1; k < offspring[i].numInstances(); k++) {

                    if (Math.abs(m_distFunc.distance(offspring[i].get(j), offspring[i].get(k))) <= m_duplicateThreshold) {
                        //toRemove.add(k);
                        if(offspring[i].numInstances() > 2) 
                            offspring[i].remove(k);
                        else {
                            //mutate to avoid having only one centroid 

                            while(Math.abs(m_distFunc.distance(offspring[i].get(j), offspring[i].get(k))) <= m_duplicateThreshold) {
                                //mutate
                                int attribute = m_rand.nextInt(offspring[i].numAttributes());
                                double val;

                                do {

                                    if (offspring[i].attribute(attribute).isNumeric()) {
                                        val = m_data.attributeStats(attribute).numericStats.min + (m_rand.nextDouble() * ((m_data.attributeStats(attribute).numericStats.max - m_data.attributeStats(attribute).numericStats.min) + 1));
                                        while (val == Double.POSITIVE_INFINITY) {
                                            val = m_data.attributeStats(attribute).numericStats.min + (m_rand.nextDouble() * ((m_data.attributeStats(attribute).numericStats.max - m_data.attributeStats(attribute).numericStats.min) + 1));
                                        }
                                    } else {
                                        val = (double) m_rand.nextInt(offspring[i].attributeStats(attribute).nominalCounts.length);
                                    }

                                } while (val == offspring[i].get(k).value(attribute));

                                offspring[i].get(k).setValue(attribute, val);
                            }

                            //end the mutation to avoid only one centroid
                        }
                    }

                }

            }
//            if(toRemove.size() != 0)
//                System.out.println();

//            offspring[i].removeAll(toRemove);
            //convert to MKMeans
            try {
                MKMeans mk = new MKMeans();
                mk.setSeed(m_rand.nextInt());
                mk.setInitializationMethod(new SelectedTag(GenClustPlusPlus.SUPPLIED, GenClustPlusPlus.TAGS_SELECTION_MK));
                mk.setInitial(offspring[i]);
                mk.setDontReplaceMissingValues(m_dontReplaceMissing);
                mk.setPreserveInstancesOrder(true);
                mk.setMaxIterations(1);
                mk.buildClusterer(m_data, m_distFunc);
                offspringMK[i] = mk;
            } catch(Exception e) {
                e.printStackTrace();
            }
        }

        return offspringMK;

    }

    private MKMeans[] probabilisticCloning(MKMeans[] selectedPopulation) {

        MKMeans[] newPopulation = new MKMeans[selectedPopulation.length];

        FitnessContainer[] fArray = new FitnessContainer[selectedPopulation.length];
        double fitnessSum = 0;

        //calculate all fitnesses
        for (int i = 0; i < selectedPopulation.length; i++) {
            double fitness = fitness(selectedPopulation[i]);
            fitnessSum += fitness;
            fArray[i] = new FitnessContainer(fitness, selectedPopulation[i]);
        }

        int newCount = 0;
        while (newCount < selectedPopulation.length) {
            double p = m_rand.nextDouble();
            double cumulativeProbability = 0.0;
            for (int i = 0; i < fArray.length; i++) {
                cumulativeProbability += fArray[i].fitness / fitnessSum;
                if (p <= cumulativeProbability) { //we've selected it

                    newPopulation[newCount++] = fArray[i].clustering;
                    break;

                } //end selections
            } //end for

        }//end while

        //add mutation
        //return mutation(newPopulation);
        for (int i = 0; i < newPopulation.length; i++) {
            newPopulation[i] = mutate(newPopulation[i].getClusterCentroids());
        }
        return newPopulation;

    }

    private MKMeans[] mutation(MKMeans[] crossoverPopulation) {

        FitnessContainer[] fArray = new FitnessContainer[crossoverPopulation.length];
        double fitnessAvg = 0;
        double fitnessMax = Double.MIN_VALUE;

        //calculate all fitnesses
        for (int i = 0; i < crossoverPopulation.length; i++) {
            double fitness = fitness(crossoverPopulation[i]);
            fitnessAvg += fitness;
            fArray[i] = new FitnessContainer(fitness, crossoverPopulation[i]);
            if (fitness > fitnessMax) {
                fitnessMax = fitness;
            }
        }

        fitnessAvg /= crossoverPopulation.length;

        for (int i = 0; i < fArray.length; i++) {
            double prob = 0;
            if (fArray[i].fitness > fitnessAvg) {
                prob = (fitnessMax - fArray[i].fitness) / (2 * (fitnessMax - fitnessAvg));
            } else {
                prob = 0.5;
            }

            if (m_rand.nextDouble() <= prob) {

                //perform the mutation
                Instances centroids = fArray[i].clustering.getClusterCentroids();
                crossoverPopulation[i] = mutate(centroids);
                
                //System.out.println("Fitness after mutation: " + fitness(crossoverPopulation[i]) + " / " + fArray[i].fitness + "\n\n");

            }

        }

        return crossoverPopulation;

    }

    private MKMeans mutate(Instances centroids) {

        //perform the mutation
        for (int j = 0; j < centroids.numInstances(); j++) {
            //change a random attribute to a random value
            int attribute = m_rand.nextInt(centroids.numAttributes());
            double val;

            do {
                if (centroids.attribute(attribute).isNumeric()) {
                    val = m_data.attributeStats(attribute).numericStats.min + (m_rand.nextDouble() * ((m_data.attributeStats(attribute).numericStats.max - m_data.attributeStats(attribute).numericStats.min) + 1));
                    while (val == Double.POSITIVE_INFINITY) {
                        val = m_data.attributeStats(attribute).numericStats.min + (m_rand.nextDouble() * ((m_data.attributeStats(attribute).numericStats.max - m_data.attributeStats(attribute).numericStats.min) + 1));
                    }
                } else {
                    val = (double) m_rand.nextInt(centroids.attributeStats(attribute).nominalCounts.length);
                }

            } while (val == centroids.get(j).value(attribute));

            centroids.get(j).setValue(attribute, val);

        } //end gene loop
        
        //duplicate removal

        for (int j = 0; j < centroids.numInstances(); j++) {

            for (int k = j + 1; k < centroids.numInstances(); k++) {

                if (Math.abs(m_distFunc.distance(centroids.get(j), centroids.get(k))) <= m_duplicateThreshold) {
                    if(centroids.numInstances() > 2)
                        centroids.remove(k);
                    else {
                        
                        //mutate to avoid having only one centroid 
                        
                        while(Math.abs(m_distFunc.distance(centroids.get(j), centroids.get(k))) <= m_duplicateThreshold) {
                            //mutate
                            int attribute = m_rand.nextInt(centroids.numAttributes());
                            double val;
                            
                            do {

                                if (centroids.attribute(attribute).isNumeric()) {
                                    val = m_data.attributeStats(attribute).numericStats.min + (m_rand.nextDouble() * ((m_data.attributeStats(attribute).numericStats.max - m_data.attributeStats(attribute).numericStats.min) + 1));
                                    while (val == Double.POSITIVE_INFINITY) {
                                        val = m_data.attributeStats(attribute).numericStats.min + (m_rand.nextDouble() * ((m_data.attributeStats(attribute).numericStats.max - m_data.attributeStats(attribute).numericStats.min) + 1));
                                    }
                                } else {
                                    val = (double) m_rand.nextInt(centroids.attributeStats(attribute).nominalCounts.length);
                                }

                            } while (val == centroids.get(k).value(attribute));

                            centroids.get(k).setValue(attribute, val);
                        }
                        
                        //end the mutation to avoid only one centroid
                        
                    }
                }

            }

        } //end duplicate removal

        try 
        {
            MKMeans mk = new MKMeans();
            mk.setSeed(m_rand.nextInt());
            mk.setInitializationMethod(new SelectedTag(GenClustPlusPlus.SUPPLIED, GenClustPlusPlus.TAGS_SELECTION_MK));
            mk.setInitial(centroids);
            mk.setMaxIterations(1);
            mk.setDontReplaceMissingValues(m_dontReplaceMissing);
            mk.setPreserveInstancesOrder(true);
            mk.buildClusterer(m_data, m_distFunc);
            return mk;
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        
        return null;
               
    }

    private MKMeans[] elitism(MKMeans[] population) throws Exception {
        //Perform elitism
        double worstFitness = Double.MAX_VALUE;
        int worstIndex = Integer.MAX_VALUE;
        double newBestFitness = Double.MIN_VALUE;
        int newBestIndex = Integer.MAX_VALUE;
        for (int i = 0; i < population.length; i++) {

            double f = fitness(population[i]);
            if (f < worstFitness) {
                worstFitness = f;
                worstIndex = i;
            }
            if (f > newBestFitness) {
                newBestFitness = f;
                newBestIndex = i;
            }
        }
        if (m_bestFitness > worstFitness) {
            population[worstIndex] = new MKMeans(m_bestChromosome);
        }
        if (newBestFitness > m_bestFitness) {
            if (newBestFitness == Double.POSITIVE_INFINITY) {
                MKMeans finalRun = new MKMeans();
                finalRun.setSeed(m_rand.nextInt());
                finalRun.setInitializationMethod(new SelectedTag(GenClustPlusPlus.SUPPLIED, GenClustPlusPlus.TAGS_SELECTION_MK));
                finalRun.setInitial(population[newBestIndex].getClusterCentroids());
                finalRun.setDontReplaceMissingValues(m_dontReplaceMissing);
                finalRun.setPreserveInstancesOrder(true);
                finalRun.setMaxIterations(m_maxKMeansIterationsFinal);
                finalRun.buildClusterer(m_data, m_distFunc);
                m_builtClusterer = finalRun;
                m_numberOfClusters = m_builtClusterer.getClusterCentroids().size();
            }
            m_bestChromosome = new MKMeans(population[newBestIndex]);
            m_bestFitness = newBestFitness;
        }
        return population;

    }

    private class FitnessContainer implements Comparable<FitnessContainer> {

        double fitness;
        MKMeans clustering;

        FitnessContainer(double f, MKMeans c) {
            fitness = f;
            clustering = c;
        }

        @Override
        public int compareTo(FitnessContainer other) {
            return Double.compare(fitness, other.fitness);
        }

    }

    /**
     * Clusters a record with the internal pre-built MKMeans
     *
     * @param instance - Record to Cluster
     * @return id of cluster assigned to instance
     * @throws Exception if instance could not be classified successfully
     */
    @Override
    public int clusterInstance(Instance instance) throws Exception {
       
        return m_builtClusterer.clusterInstance(instance);

    }

    /**
     * Return a string describing this clusterer.
     *
     * @return a description of the clusterer as a string
     */
    @Override
    public String toString() {
        if (m_builtClusterer == null) {
            return "No clusterer built yet!";
        }
        return m_builtClusterer.toString();
    }

    /**
     * Returns default capabilities of the clusterer.
     *
     * @return the capabilities of this clusterer
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NO_CLASS);

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        return result;
    }

    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    public String globalInfo() {
        return "Class implementing algorithm described in \"Combining K-Means and a "
                + "Genetic Algorithm through a Novel Arrangement of Genetic "
                + "Operators for High Quality Clustering\".\n\n"
                + "Differences to the original algorithm are: \n"
                + "1. No use of VICUS similarity measure - standard ManhattanDistance"
                + "class is used instead.\n"
                + "2. Uses the basic missing value handling from SimpleKMeans.\n"
                + "3. If an operation generates a chromosome where all records are "
                + "assigned to a single cluster, chromosome will be mutated until at"
                + " least 2 clusters are found.\n"
                + "4. The starting generation for the chromosome selection "
                + "operation is now modifiable, where in the original paper"
                + "it was set to 11. The default is now after generation 50 ("
                + "with the default number of generations being 60).\n\n"
                + "For more information see:" + getTechnicalInformation().toString();
    }

    /**
     * Main method for executing this class.
     *
     * @param args
     */
    public static void main(String[] args)  throws Exception{
        runClusterer(new GenClustPlusPlus(), args);        
    }

    /**
     * Set options for clusterer
     * @param options
     * @throws Exception if setting an option is unsuccessful
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        m_dontReplaceMissing = Utils.getFlag("M", options);

        String optionString = Utils.getOption("I", options);
        if (optionString.length() != 0) {
            setNumGenerations(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("P", options);
        if (optionString.length() != 0) {
            setInitialPopulationSize(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("N", options);
        if (optionString.length() != 0) {
            setMaxKMeansIterationsInitial(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("Q", options);
        if (optionString.length() != 0) {
            setMaxKMeansIterationsQuick(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("F", options);
        if (optionString.length() != 0) {
            setMaxKMeansIterationsFinal(Integer.parseInt(optionString));
        }
        
        optionString = Utils.getOption("C", options);
        if (optionString.length() != 0) {
            setStartChromosomeSelectionGeneration(Integer.parseInt(optionString));
        }
        

        optionString = Utils.getOption("D", options);
        if (optionString.length() != 0) {
            setDuplicateThreshold(Double.parseDouble(optionString));
        }
        
        optionString = Utils.getOption("S", options);
        if (optionString.length() != 0) {
            setSeed(Integer.parseInt(optionString));
        }


    }

    /**
     * Gets the current settings of GenClust++.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-I");
        result.add("" + getNumGenerations());

        result.add("-P");
        result.add("" + getInitialPopulationSize());

        result.add("-N");
        result.add("" + getMaxKMeansIterationsInitial());

        result.add("-Q");
        result.add("" + getMaxKMeansIterationsQuick());

        result.add("-F");
        result.add("" + getMaxKMeansIterationsFinal());

        result.add("-D");
        result.add("" + getDuplicateThreshold());
        
        result.add("-C");
        result.add("" + getStartChromosomeSelectionGeneration());

        result.add("-S");
        result.add("" + getSeed());

        if (m_dontReplaceMissing) {
            result.add("-M");
        }

        return result.toArray(new String[result.size()]);
    }

    /**
     * set the number of generations to be performed for genetic algoruthm.
     *
     * @param n the number of generations
     * @throws Exception if number of generations is smaller than 1
     */
    public void setNumGenerations(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Number of generations must be > 0");
        }
        m_numberOfGenerations = n;
    }

    /**
     * set the initial population size.
     *
     * @param n the initial population size
     * @throws Exception if inital population size is smaller than 3
     */
    public void setInitialPopulationSize(int n) throws Exception {
        if (n <= 2) {
            throw new Exception("Initial population must be > 2");
        }
        if ((n % 2) != 0) {
            throw new Exception("Initial population must be divisible by 2");
        }
        m_initialPopulationSize = n;
    }

    /**
     * set the max initial k means iterations.
     *
     * @param n the max initial k means iterations
     * @throws Exception if max initial k means iterations is non-positive
     */
    public void setMaxKMeansIterationsInitial(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Max initial k means iterations must be positive.");
        }
        m_maxKMeansIterationsInitial = n;
    }

    /**
     * set the max quick k means iterations.
     *
     * @param n the max quick k means iterations
     * @throws Exception if max quick k means iterations is non-positive
     */
    public void setMaxKMeansIterationsQuick(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Max quick k means iterations must be positive.");
        }
        m_maxKMeansIterationsQuick = n;
    }

    /**
     * set the max final k means iterations.
     *
     * @param n the max final k means iterations
     * @throws Exception if max final k means iterations is non-positive
     */
    public void setMaxKMeansIterationsFinal(int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Max final k means iterations must be positive.");
        }
        m_maxKMeansIterationsFinal = n;
    }

    /**
     * set difference threshold for determining duplicate genomes.
     *
     * @param d - the new difference threshold for determining duplicate genomes
     * @throws Exception if d not between 0 and 1
     */
    public void setDuplicateThreshold(double d) throws Exception {
        if (d < 0 || d > 1) {
            throw new Exception("Duplicate threshold must be between 0 and 1");
        }
        m_duplicateThreshold = d;
    }

    /**
     * By default, missing values are replaced with global mean / mode. Setting
     * this to true disables this basic imputation.
     *
     * @param m_dontReplaceMissing
     */
    public void setDontReplaceMissing(boolean m_dontReplaceMissing) {
        this.m_dontReplaceMissing = m_dontReplaceMissing;
    }
    
    /**
     * Set generation after which to start chromosome slection.
     *
     * @param s - the new starting generation
     * @throws Exception if s < 1
     */
    public void setStartChromosomeSelectionGeneration(int s) throws Exception {
        if (s < 1) {
            throw new Exception("Chromosome Selection start generation must be "
                    + "greater than 0");
        }
        m_startChromosomeSelectionGeneration = s;
    }

    /**
     * Returns the number of generations for the genetic algorithm
     *
     * @return number of generations to be performed
     */
    public int getNumGenerations() {
        return m_numberOfGenerations;
    }
    
    /**
     * Returns the generation after which to start using the chromosome selection 
     * operation. Cannot be less than 1.
     *
     * @return starting generation for chromosome selection
     */
    public int getStartChromosomeSelectionGeneration() {
        return m_startChromosomeSelectionGeneration;
    }

    /**
     * Returns the initial population size
     *
     * @return initial population size
     */
    public int getInitialPopulationSize() {
        return m_initialPopulationSize;
    }

    /**
     * Returns the max iterations for initial k-means
     *
     * @return max iterations for initial k-means
     */
    public int getMaxKMeansIterationsInitial() {
        return m_maxKMeansIterationsInitial;
    }

    /**
     * Returns the max iterations for quick k-means
     *
     * @return max iterations for quick k-means
     */
    public int getMaxKMeansIterationsQuick() {
        return m_maxKMeansIterationsQuick;
    }

    /**
     * Returns the max iterations for final k-means
     *
     * @return max iterations for final k-means
     */
    public int getMaxKMeansIterationsFinal() {
        return m_maxKMeansIterationsFinal;
    }

    /**
     * Returns the duplicate threshold
     *
     * @return duplicate threshold
     */
    public double getDuplicateThreshold() {
        return m_duplicateThreshold;
    }

    /**
     * Return whether to replace missing values or not.
     *
     * @return whether or not to replace missing values
     */
    public boolean getDontReplaceMissing() {
        return m_dontReplaceMissing;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String numGenerationsTipText() {
        return "set number of genetic algorithm generations";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String initialPopulationSizeTipText() {
        return "set initial population size for genetic algorithm";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String maxKMeansIterationsInitialTipText() {
        return "set the max iterations for initial k-means";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String maxKMeansIterationsQuickTipText() {
        return "set the max iterations for quick k-means";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String maxKMeansIterationsFinalTipText() {
        return "set the max iterations for final k-means";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String duplicateThresholdTipText() {
        return "set the duplicate threshold";
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String dontReplaceMissingTipText() {
        return "Replace missing values globally with mean/mode.";
    }
    
    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String startChromosomeSelectionGenerationTipText() {
        return "Generation after which to start using the chromosome selection "
                + "operation.";
    }

    class MKMeans extends SimpleKMeans {

        private static final long serialVersionUID = -2890080669539418269L;
        protected boolean m_supplied = false;
        protected boolean m_unable = false;

        public MKMeans() {
            super();
        }

        public MKMeans(MKMeans mk) throws Exception {
            super();
            m_ReplaceMissingFilter = new ReplaceMissingValues();
            m_ReplaceMissingFilter.setInputFormat(mk.m_ReplaceMissingFilter.getCopyOfInputFormat());
            this.setPreserveInstancesOrder(true);
            this.m_ClusterCentroids = new Instances(mk.getClusterCentroids());
            this.m_NumClusters = this.m_ClusterCentroids.numInstances();
            this.m_Assignments = new int[mk.getAssignments().length];
            this.m_DistanceFunction = mk.m_DistanceFunction;
            this.m_dontReplaceMissing = mk.getDontReplaceMissingValues();
            System.arraycopy(mk.getAssignments(), 0, m_Assignments, 0, mk.getAssignments().length);
        }

        /**
         * Generates a clusterer. Has to initialize all fields of the clusterer
         * that are not being set via options.
         *
         * @param data set of instances serving as training data
         * @throws Exception if the clusterer has not been generated
         * successfully
         */
        @Override
        public void buildClusterer(Instances data) throws Exception {

            //this.m_DistanceFunction = new MKMeansDistance(data);
            this.m_DistanceFunction = new ManhattanDistance(data);
            super.buildClusterer(data);

        }

        public void buildClusterer(Instances data, ManhattanDistance distance) throws Exception {

            this.m_DistanceFunction = distance;
            m_canopyClusters = null;

            // can clusterer handle the data?
            getCapabilities().testWithFail(data);

            m_Iterations = 0;

            m_ReplaceMissingFilter = new ReplaceMissingValues();
            Instances instances = new Instances(data);

            instances.setClassIndex(-1);
            if (!m_dontReplaceMissing) {
                m_ReplaceMissingFilter.setInputFormat(instances);
//                instances = Filter.useFilter(instances, m_ReplaceMissingFilter);
            }

            m_ClusterNominalCounts = new double[m_NumClusters][instances.numAttributes()][];
            m_ClusterMissingCounts = new double[m_NumClusters][instances.numAttributes()];
            if (m_displayStdDevs) {
                m_FullStdDevs = instances.variances();
            }

            m_FullMeansOrMediansOrModes = moveCentroid(0, instances, true, false);

            m_FullMissingCounts = m_ClusterMissingCounts[0];
            m_FullNominalCounts = m_ClusterNominalCounts[0];
            double sumOfWeights = instances.sumOfWeights();
            for (int i = 0; i < instances.numAttributes(); i++) {
                if (instances.attribute(i).isNumeric()) {
                    if (m_displayStdDevs) {
                        m_FullStdDevs[i] = Math.sqrt(m_FullStdDevs[i]);
                    }
                    if (m_FullMissingCounts[i] == sumOfWeights) {
                        m_FullMeansOrMediansOrModes[i] = Double.NaN; // mark missing as mean
                    }
                } else if (m_FullMissingCounts[i] > m_FullNominalCounts[i][Utils
                        .maxIndex(m_FullNominalCounts[i])]) {
                    m_FullMeansOrMediansOrModes[i] = -1; // mark missing as most common
                    // value
                }
            }

            m_ClusterCentroids = new Instances(instances, m_NumClusters);
            int[] clusterAssignments = new int[instances.numInstances()];

            if (m_PreserveOrder) {
                m_Assignments = clusterAssignments;
            }

            m_DistanceFunction.setInstances(instances);

            Random RandomO = new Random(getSeed());
            int instIndex;
            HashMap<DecisionTableHashKey, Integer> initC
                    = new HashMap<DecisionTableHashKey, Integer>();
            DecisionTableHashKey hk = null;

            Instances initInstances = null;
            if (m_PreserveOrder) {
                initInstances = new Instances(instances);
            } else {
                initInstances = instances;
            }

            if (m_speedUpDistanceCompWithCanopies) {
                m_canopyClusters = new Canopy();
                m_canopyClusters.setNumClusters(m_NumClusters);
                m_canopyClusters.setSeed(getSeed());
                m_canopyClusters.setT2(getCanopyT2());
                m_canopyClusters.setT1(getCanopyT1());
                m_canopyClusters
                        .setMaxNumCandidateCanopiesToHoldInMemory(getCanopyMaxNumCanopiesToHoldInMemory());
                m_canopyClusters.setPeriodicPruningRate(getCanopyPeriodicPruningRate());
                m_canopyClusters.setMinimumCanopyDensity(getCanopyMinimumCanopyDensity());
                m_canopyClusters.setDebug(getDebug());
                m_canopyClusters.buildClusterer(initInstances);
                // System.err.println(m_canopyClusters);
                m_centroidCanopyAssignments = new ArrayList<long[]>();
                m_dataPointCanopyAssignments = new ArrayList<long[]>();
            }

            if (m_initializationMethod == KMEANS_PLUS_PLUS) {
                kMeansPlusPlusInit(initInstances);

                m_initialStartPoints = new Instances(m_ClusterCentroids);
            } else if (m_initializationMethod == CANOPY) {
                canopyInit(initInstances);

                m_initialStartPoints = new Instances(m_canopyClusters.getCanopies());
            } else if (m_initializationMethod == FARTHEST_FIRST) {
                farthestFirstInit(initInstances);

                m_initialStartPoints = new Instances(m_ClusterCentroids);
            } else if (m_initializationMethod == SUPPLIED || m_supplied) {

                m_ClusterCentroids = m_initialStartPoints;
                m_NumClusters = m_initialStartPoints.numInstances();

                if (!m_supplied) {
                    throw new Exception("Please supply a set of initial centroids.");
                }

            } else {
                // random
                for (int j = initInstances.numInstances() - 1; j >= 0; j--) {
                    instIndex = RandomO.nextInt(j + 1);
                    hk
                            = new DecisionTableHashKey(initInstances.instance(instIndex),
                                    initInstances.numAttributes(), true);
                    if (!initC.containsKey(hk)) {
                        m_ClusterCentroids.add(initInstances.instance(instIndex));
                        initC.put(hk, null);
                    }
                    initInstances.swap(j, instIndex);

                    if (m_ClusterCentroids.numInstances() == m_NumClusters) {
                        break;
                    }
                }

                m_initialStartPoints = new Instances(m_ClusterCentroids);
            }

            if (m_speedUpDistanceCompWithCanopies) {
                // assign canopies to training data
                for (int i = 0; i < instances.numInstances(); i++) {
                    m_dataPointCanopyAssignments.add(m_canopyClusters
                            .assignCanopies(instances.instance(i)));
                }
            }

            m_NumClusters = m_ClusterCentroids.numInstances();

            // removing reference
            initInstances = null;

            int i;
            boolean converged = false;
            int emptyClusterCount;
            Instances[] tempI = new Instances[m_NumClusters];
            m_squaredErrors = new double[m_NumClusters];
            m_ClusterNominalCounts = new double[m_NumClusters][instances.numAttributes()][0];
            m_ClusterMissingCounts = new double[m_NumClusters][instances.numAttributes()];
            startExecutorPool();

            while (!converged) {
                if (m_speedUpDistanceCompWithCanopies) {
                    // re-assign canopies to the current cluster centers
                    m_centroidCanopyAssignments.clear();
                    for (int kk = 0; kk < m_ClusterCentroids.numInstances(); kk++) {
                        m_centroidCanopyAssignments.add(m_canopyClusters
                                .assignCanopies(m_ClusterCentroids.instance(kk)));
                    }
                }

                emptyClusterCount = 0;
                m_Iterations++;
                converged = true;

                if (m_executionSlots <= 1
                        || instances.numInstances() < 2 * m_executionSlots) {
                    for (i = 0; i < instances.numInstances(); i++) {
                        Instance toCluster = instances.instance(i);
                        int newC
                                = clusterProcessedInstance(
                                        toCluster,
                                        false,
                                        false,
                                        m_speedUpDistanceCompWithCanopies ? m_dataPointCanopyAssignments
                                                        .get(i) : null);
                        if (newC != clusterAssignments[i]) {
                            converged = false;
                        }
                        clusterAssignments[i] = newC;
                    }
                } else {
                    converged = launchAssignToClusters(instances, clusterAssignments);
                }

                m_ClusterCentroids = new Instances(instances, m_NumClusters);
                for (i = 0; i < m_NumClusters; i++) {
                    tempI[i] = new Instances(instances, 0);
                }
                for (i = 0; i < instances.numInstances(); i++) {
                    tempI[clusterAssignments[i]].add(instances.instance(i));
                }
                if (m_initializationMethod == SUPPLIED && m_MaxIterations == 1) { //nothing
                    m_ClusterCentroids = m_initialStartPoints;
                } else {
                    if (m_executionSlots <= 1
                            || instances.numInstances() < 2 * m_executionSlots) {
                        for (i = 0; i < m_NumClusters; i++) {
                            if (tempI[i].numInstances() == 0) {
                                // empty cluster
                                emptyClusterCount++;
                            } else {
                                moveCentroid(i, tempI[i], true, true);
                            }
                        }
                    } else {
                        emptyClusterCount = launchMoveCentroids(tempI);
                    }
                }

                if (m_Iterations == m_MaxIterations) {
                    converged = true;
                }

//                if (emptyClusterCount > 0) {
//                    m_NumClusters -= emptyClusterCount;
//                    if (converged) {
//                        Instances[] t = new Instances[m_NumClusters];
//                        int index = 0;
//                        for (int k = 0; k < tempI.length; k++) {
//                            if (tempI[k].numInstances() > 0) {
//                                t[index] = tempI[k];
//
//                                for (i = 0; i < tempI[k].numAttributes(); i++) {
//                                    m_ClusterNominalCounts[index][i] = m_ClusterNominalCounts[k][i];
//                                }
//                                index++;
//                            }
//                        }
//                        tempI = t;
//
//                    } else {
//                        tempI = new Instances[m_NumClusters];
//                    }
//
//                }

                if (!converged) {
                    m_ClusterNominalCounts = new double[m_NumClusters][instances.numAttributes()][0];
                }
            }

            for (i = 0; i < instances.numInstances(); i++) {
                Instance toCluster = instances.instance(i);
                int newC
                        = clusterProcessedInstance(
                                toCluster,
                                false,
                                true,
                                m_speedUpDistanceCompWithCanopies ? m_dataPointCanopyAssignments
                                                .get(i) : null);
                clusterAssignments[i] = newC;
            }
            m_Assignments = clusterAssignments;

            // calculate errors
            if (!m_FastDistanceCalc) {
                for (i = 0; i < instances.numInstances(); i++) {
                    clusterProcessedInstance(instances.instance(i), true, false, null);
                }
            }

            if (m_displayStdDevs) {
                m_ClusterStdDevs = new Instances(instances, m_NumClusters);
            }
            m_ClusterSizes = new double[m_NumClusters];
            for (i = 0; i < m_NumClusters; i++) {
                if (m_displayStdDevs) {
                    double[] vals2 = tempI[i].variances();
                    for (int j = 0; j < instances.numAttributes(); j++) {
                        if (instances.attribute(j).isNumeric()) {
                            vals2[j] = Math.sqrt(vals2[j]);
                        } else {
                            vals2[j] = Utils.missingValue();
                        }
                    }
                    m_ClusterStdDevs.add(new DenseInstance(1.0, vals2));
                }
                m_ClusterSizes[i] = tempI[i].sumOfWeights();
            }

            m_executorPool.shutdown();

            // save memory!
            //m_DistanceFunction.clean();
            m_NumClusters = m_ClusterCentroids.numInstances();

        }

        /**
         * sets the distance function to use for instance comparison.
         *
         * @param df the new distance function to use
         * @throws Exception if instances cannot be processed
         */
        @Override
        public void setDistanceFunction(DistanceFunction df) throws Exception {
            /*if (!(df instanceof MKMeansDistance)) {
            throw new Exception(
                    "MKMeans Only Supports MKMeansDistance - use the MKMeans(Instances data) constructor.");
        }*/
            m_DistanceFunction = df;
        }

        /**
         * clusters an instance that has been through the filters.
         *
         * @param instance the instance to assign a cluster to
         * @param updateErrors if true, update the within clusters sum of errors
         * @param useFastDistCalc whether to use the fast distance calculation
         * or not
         * @param instanceCanopies the canopies covering the instance to be
         * clustered, or null if not using the option to reduce the number of
         * distance computations via canopies
         * @return a cluster number
         */
        private int clusterProcessedInstance(Instance instance, boolean updateErrors,
                boolean useFastDistCalc, long[] instanceCanopies) {
            double minDist = Integer.MAX_VALUE;
            int bestCluster = 0;

            for (int i = 0; i < m_ClusterCentroids.numInstances(); i++) {
                double dist;
                if (useFastDistCalc) {
                    if (m_speedUpDistanceCompWithCanopies && instanceCanopies != null
                            && instanceCanopies.length > 0) {
                        try {
                            if (!Canopy.nonEmptyCanopySetIntersection(
                                    m_centroidCanopyAssignments.get(i), instanceCanopies)) {
                                continue;
                            }
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                        dist
                                = m_DistanceFunction.distance(instance,
                                        m_ClusterCentroids.instance(i), minDist);
                    } else {
                        dist
                                = m_DistanceFunction.distance(instance,
                                        m_ClusterCentroids.instance(i), minDist);
                    }
                } else {
                    dist
                            = m_DistanceFunction.distance(instance, m_ClusterCentroids.instance(i));
                }
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = i;
                }
            }
            if (updateErrors) {
                if (m_DistanceFunction instanceof EuclideanDistance) {
                    // Euclidean distance to Squared Euclidean distance
                    minDist *= minDist * instance.weight();
                }
                m_squaredErrors[bestCluster] += minDist;
            }
            return bestCluster;
        }

        public void setInitial(Instances initial) {
            m_supplied = true;
            m_NumClusters = initial.numInstances();
            m_initialStartPoints = initial;
        }

        /* Set the initialization method to use
    * 
    * @param method the initialization method to use
         */
        @Override
        public void setInitializationMethod(SelectedTag method) {
            if (method.getTags() == TAGS_SELECTION_MK) {
                m_initializationMethod = method.getSelectedTag().getID();
            }
        }

        @Override
        public int clusterInstance(Instance instance) throws Exception {
            Instance inst = null;
            if (!m_dontReplaceMissing) {
              m_ReplaceMissingFilter.input(instance);
              m_ReplaceMissingFilter.batchFinished();
              inst = m_ReplaceMissingFilter.output();
            } else {
              inst = instance;
            }
            return clusterProcessedInstance(inst, false, false, null);
        }

        /**
         * return a string describing this clusterer.
         *
         * @return a description of the clusterer as a string
         */
        @Override
        public String toString() {
            if (m_ClusterCentroids == null) {
                return "No clusterer built yet!";
            }

            int maxWidth = 0;
            int maxAttWidth = 0;
            boolean containsNumeric = false;
            for (int i = 0; i < m_NumClusters; i++) {
                for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
                    if (m_ClusterCentroids.attribute(j).name().length() > maxAttWidth) {
                        maxAttWidth = m_ClusterCentroids.attribute(j).name().length();
                    }
                    if (m_ClusterCentroids.attribute(j).isNumeric()) {
                        containsNumeric = true;
                        double width
                                = Math.log(Math.abs(m_ClusterCentroids.instance(i).value(j)))
                                / Math.log(10.0);

                        if (width < 0) {
                            width = 1;
                        }
                        // decimal + # decimal places + 1
                        width += 6.0;
                        if ((int) width > maxWidth) {
                            maxWidth = (int) width;
                        }
                    }
                }
            }

            for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
                if (m_ClusterCentroids.attribute(i).isNominal()) {
                    Attribute a = m_ClusterCentroids.attribute(i);
                    for (int j = 0; j < m_ClusterCentroids.numInstances(); j++) {
                        String val = a.value((int) m_ClusterCentroids.instance(j).value(i));
                        if (val.length() > maxWidth) {
                            maxWidth = val.length();
                        }
                    }
                    for (int j = 0; j < a.numValues(); j++) {
                        String val = a.value(j) + " ";
                        if (val.length() > maxAttWidth) {
                            maxAttWidth = val.length();
                        }
                    }
                }
            }

            if (m_displayStdDevs) {
                // check for maximum width of maximum frequency count
                for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
                    if (m_ClusterCentroids.attribute(i).isNominal()) {
                        int maxV = Utils.maxIndex(m_FullNominalCounts[i]);
                        /*
           * int percent = (int)((double)m_FullNominalCounts[i][maxV] /
           * Utils.sum(m_ClusterSizes) * 100.0);
                         */
                        int percent = 6; // max percent width (100%)
                        String nomV = "" + m_FullNominalCounts[i][maxV];
                        // + " (" + percent + "%)";
                        if (nomV.length() + percent > maxWidth) {
                            maxWidth = nomV.length() + 1;
                        }
                    }
                }
            }

            // check for size of cluster sizes
            for (double m_ClusterSize : m_ClusterSizes) {
                String size = "(" + m_ClusterSize + ")";
                if (size.length() > maxWidth) {
                    maxWidth = size.length();
                }
            }

            if (m_displayStdDevs && maxAttWidth < "missing".length()) {
                maxAttWidth = "missing".length();
            }

            String plusMinus = "+/-";
            maxAttWidth += 2;
            if (m_displayStdDevs && containsNumeric) {
                maxWidth += plusMinus.length();
            }
            if (maxAttWidth < "Attribute".length() + 2) {
                maxAttWidth = "Attribute".length() + 2;
            }

            if (maxWidth < "Full Data".length()) {
                maxWidth = "Full Data".length() + 1;
            }

            if (maxWidth < "missing".length()) {
                maxWidth = "missing".length() + 1;
            }

            StringBuffer temp = new StringBuffer();
            if(!m_unable)
                temp.append("\nkMeans after final generation\n======\n");
            else
                temp.append("\nSimpleKMeans Results\n======\n");
            temp.append("\nNumber of iterations: " + m_Iterations);

            if (!m_FastDistanceCalc) {
                temp.append("\n");
                if (m_DistanceFunction instanceof EuclideanDistance) {
                    temp.append("Within cluster sum of squared errors: "
                            + Utils.sum(m_squaredErrors));
                } else {
                    temp.append("Sum of within cluster distances: "
                            + Utils.sum(m_squaredErrors));
                }
            }

            temp.append("\n\nInitial starting points");
            if(!m_unable)
                temp.append(" (final chromosome)");
            else
                temp.append("");
            temp.append(":\n");
            if (m_initializationMethod != CANOPY) {
                temp.append("\n");
                for (int i = 0; i < m_initialStartPoints.numInstances(); i++) {
                    temp.append("Cluster " + i + ": " + m_initialStartPoints.instance(i))
                            .append("\n");
                }
            } else {
                temp.append(m_canopyClusters.toString(false));
            }

            if (m_speedUpDistanceCompWithCanopies) {
                temp
                        .append("\nReduced number of distance calculations by using canopies.");
                if (m_initializationMethod != CANOPY) {
                    temp.append("\nCanopy T2 radius: "
                            + String.format("%-10.3f", m_canopyClusters.getActualT2()));
                    temp.append(
                            "\nCanopy T1 radius: "
                            + String.format("%-10.3f", m_canopyClusters.getActualT1())).append(
                            "\n");
                }
            }

            if (!m_dontReplaceMissing) {
                temp.append("\nMissing values globally replaced with mean/mode");
            }

            temp.append("\n\nFinal cluster centroids:\n");
            temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
                    - "Cluster#".length(), true));

            temp.append("\n");
            temp
                    .append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

            temp
                    .append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

            // cluster numbers
            for (int i = 0; i < m_NumClusters; i++) {
                String clustNum = "" + i;
                temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
            }
            temp.append("\n");

            // cluster sizes
            String cSize = "(" + Utils.sum(m_ClusterSizes) + ")";
            temp.append(pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(),
                    true));
            for (int i = 0; i < m_NumClusters; i++) {
                cSize = "(" + m_ClusterSizes[i] + ")";
                temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
            }
            temp.append("\n");

            temp.append(pad("", "=",
                    maxAttWidth
                    + (maxWidth * (m_ClusterCentroids.numInstances() + 1)
                    + m_ClusterCentroids.numInstances() + 1), true));
            temp.append("\n");

            for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
                String attName = m_ClusterCentroids.attribute(i).name();
                temp.append(attName);
                for (int j = 0; j < maxAttWidth - attName.length(); j++) {
                    temp.append(" ");
                }

                String strVal;
                String valMeanMode;
                // full data
                if (m_ClusterCentroids.attribute(i).isNominal()) {
                    if (m_FullMeansOrMediansOrModes[i] == -1) { // missing
                        valMeanMode
                                = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode
                                = pad(
                                        (strVal
                                        = m_ClusterCentroids.attribute(i).value(
                                                (int) m_FullMeansOrMediansOrModes[i])), " ", maxWidth + 1
                                        - strVal.length(), true);
                    }
                } else {
                    if (Double.isNaN(m_FullMeansOrMediansOrModes[i])) {
                        valMeanMode
                                = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode
                                = pad(
                                        (strVal
                                        = Utils.doubleToString(m_FullMeansOrMediansOrModes[i], maxWidth,
                                                4).trim()), " ", maxWidth + 1 - strVal.length(), true);
                    }
                }
                temp.append(valMeanMode);

                for (int j = 0; j < m_NumClusters; j++) {
                    if (m_ClusterCentroids.attribute(i).isNominal()) {
                        if (m_ClusterCentroids.instance(j).isMissing(i)) {
                            valMeanMode
                                    = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                        } else {
                            valMeanMode
                                    = pad(
                                            (strVal
                                            = m_ClusterCentroids.attribute(i).value(
                                                    (int) m_ClusterCentroids.instance(j).value(i))), " ",
                                            maxWidth + 1 - strVal.length(), true);
                        }
                    } else {
                        if (m_ClusterCentroids.instance(j).isMissing(i)) {
                            valMeanMode
                                    = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                        } else {
                            valMeanMode
                                    = pad(
                                            (strVal
                                            = Utils.doubleToString(m_ClusterCentroids.instance(j).value(i),
                                                    maxWidth, 4).trim()), " ", maxWidth + 1 - strVal.length(),
                                            true);
                        }
                    }
                    temp.append(valMeanMode);
                }
                temp.append("\n");

                if (m_displayStdDevs) {
                    // Std devs/max nominal
                    String stdDevVal = "";

                    if (m_ClusterCentroids.attribute(i).isNominal()) {
                        // Do the values of the nominal attribute
                        Attribute a = m_ClusterCentroids.attribute(i);
                        for (int j = 0; j < a.numValues(); j++) {
                            // full data
                            String val = "  " + a.value(j);
                            temp.append(pad(val, " ", maxAttWidth + 1 - val.length(), false));
                            double count = m_FullNominalCounts[i][j];
                            int percent
                                    = (int) ((double) m_FullNominalCounts[i][j]
                                    / Utils.sum(m_ClusterSizes) * 100.0);
                            String percentS = "" + percent + "%)";
                            percentS = pad(percentS, " ", 5 - percentS.length(), true);
                            stdDevVal = "" + count + " (" + percentS;
                            stdDevVal
                                    = pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                            temp.append(stdDevVal);

                            // Clusters
                            for (int k = 0; k < m_NumClusters; k++) {
                                percent
                                        = (int) ((double) m_ClusterNominalCounts[k][i][j]
                                        / m_ClusterSizes[k] * 100.0);
                                percentS = "" + percent + "%)";
                                percentS = pad(percentS, " ", 5 - percentS.length(), true);
                                stdDevVal = "" + m_ClusterNominalCounts[k][i][j] + " (" + percentS;
                                stdDevVal
                                        = pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                                temp.append(stdDevVal);
                            }
                            temp.append("\n");
                        }
                        // missing (if any)
                        if (m_FullMissingCounts[i] > 0) {
                            // Full data
                            temp.append(pad("  missing", " ",
                                    maxAttWidth + 1 - "  missing".length(), false));
                            double count = m_FullMissingCounts[i];
                            int percent
                                    = (int) ((double) m_FullMissingCounts[i]
                                    / Utils.sum(m_ClusterSizes) * 100.0);
                            String percentS = "" + percent + "%)";
                            percentS = pad(percentS, " ", 5 - percentS.length(), true);
                            stdDevVal = "" + count + " (" + percentS;
                            stdDevVal
                                    = pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                            temp.append(stdDevVal);

                            // Clusters
                            for (int k = 0; k < m_NumClusters; k++) {
                                percent
                                        = (int) ((double) m_ClusterMissingCounts[k][i]
                                        / m_ClusterSizes[k] * 100.0);
                                percentS = "" + percent + "%)";
                                percentS = pad(percentS, " ", 5 - percentS.length(), true);
                                stdDevVal = "" + m_ClusterMissingCounts[k][i] + " (" + percentS;
                                stdDevVal
                                        = pad(stdDevVal, " ", maxWidth + 1 - stdDevVal.length(), true);
                                temp.append(stdDevVal);
                            }

                            temp.append("\n");
                        }

                        temp.append("\n");
                    } else {
                        // Full data
                        if (Double.isNaN(m_FullMeansOrMediansOrModes[i])) {
                            stdDevVal = pad("--", " ", maxAttWidth + maxWidth + 1 - 2, true);
                        } else {
                            stdDevVal
                                    = pad(
                                            (strVal
                                            = plusMinus
                                            + Utils.doubleToString(m_FullStdDevs[i], maxWidth, 4)
                                                    .trim()), " ",
                                            maxWidth + maxAttWidth + 1 - strVal.length(), true);
                        }
                        temp.append(stdDevVal);

                        // Clusters
                        for (int j = 0; j < m_NumClusters; j++) {
                            if (m_ClusterCentroids.instance(j).isMissing(i)) {
                                stdDevVal = pad("--", " ", maxWidth + 1 - 2, true);
                            } else {
                                stdDevVal
                                        = pad(
                                                (strVal
                                                = plusMinus
                                                + Utils.doubleToString(
                                                        m_ClusterStdDevs.instance(j).value(i), maxWidth, 4)
                                                        .trim()), " ", maxWidth + 1 - strVal.length(), true);
                            }
                            temp.append(stdDevVal);
                        }
                        temp.append("\n\n");
                    }
                }
            }

            temp.append("\n\n");
            return temp.toString();
        }
        
        private void setUnable(boolean u) {
            m_unable = u;
        }

        private String pad(String source, String padChar, int length, boolean leftPad) {
            StringBuffer temp = new StringBuffer();

            if (leftPad) {
                for (int i = 0; i < length; i++) {
                    temp.append(padChar);
                }
                temp.append(source);
            } else {
                temp.append(source);
                for (int i = 0; i < length; i++) {
                    temp.append(padChar);
                }
            }
            return temp.toString();
        }

    }

}
