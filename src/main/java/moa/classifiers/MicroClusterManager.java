/*
 *    RuleClassifier.java
 *    Copyright (C) 2012 
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *    
 *    
 */

package moa.classifiers;
import java.util.*; 
import java.io.*;
import java.math.BigDecimal;
import static weka.core.Utils.log2;

/**
 *
 * @author Mark Tennant @ Reading University :2015
 */


public class MicroClusterManager implements Serializable {
    private static final long serialVersionUID = 1L;
   
    private MicroCluster[][] Clusters; 
    private int[] MicroClustersCount; 
    private int[][] MicroClustersErrorCount;
    public double CF1T_Value=0;
    
    private String AdditionList = ""; 
    private String RemovalList = ""; 
          
    public int ErrorOption = 2; //1000 with RandoomTreeGenerator -- 2 with SEAGenerator -- 50 with covtypeNom and 200 window size in addition to 100000 frquency
    public int MAXClusterArraySizeOption =  100;//1000   
    public int RemovePoorlyPerformingClustersOption = 50; // Range between 0 and 100   
    public double SplitMultiAttributesOption = 50;  // Maximum number of selected attributes with high variant within split method
    public boolean SplitOnAllAttributesFlagOption = false;
    public boolean UseGlobalTimeStampsOption = true; 
    public boolean SplitClusterGetsNewTimeStampsOption =  true; 
    public boolean DangerTheoryFlagOption = true;
    public boolean DoublePunishFlagOption = true; 
    private int NumClasses = -1;
    private int NumAttributes = -1; 
          
    public String OutputPath  = "c:/";//"/home/shared/Dropbox/NETBEANS/SPARKCutdown/DATA/OUTPUT/MacroClusterDims";
       
    public boolean b_Initalised  = false;
   
    public boolean PrintStatsDebug = false;
    /**
     * Added by Mahmood
     * Saving the last values of Split and Death
     */
    public int last_SplitCounter=0;
    public int last_DeathCounter=0;
    public StringBuffer StrBuffer = new StringBuffer();
    public boolean SplitIsHappened = false;
   
    int trainingcount = 0; 
    
    // MOVED to the MC for independence testing 
    public double[] StreamSquareTotal; 
    public int[] StreamItemCount; 
    public double[] StreamTotal; 
    public double NumberSelectedFeatures = 0;
    
    public int[] trainNNIndex; 
    public double[] trainNNDistance; 
    public MicroCluster[] trainNNMicroCluster; 
     
    // stastics for De-~Bugging 
    public int SplitCount = 0;
    public int InsertCount = 0;
    public int DeleteCount = 0;
    //public int CorrectPredict = 0; 
    public int CorrectPredictTest = 0; 
    public int[][] CorrectPredictTestArray;
    
   public static class TMeasure
    {
    public double TValue = -1; 
    public int TrainingIndex = -1; 
    }
    
    public static class MyTMeasureComp implements Comparator<TMeasure>
    {
    @Override
        public int compare(TMeasure e1, TMeasure e2) 
        {
            if (e1.TValue==e2.TValue)
            {
                return 0; 
            }else{
            if(e1.TValue < e2.TValue)
            {
                return -1;
            } else {
                return 1;
            }
            }
        }
    }

     public static class MyMicroClusterNComp implements Comparator<MicroCluster>
     {
    @Override
        public int compare(MicroCluster e1, MicroCluster e2) 
        {
            if (e1.N==e2.N)
            {
                return 0; 
            }else{
            if(e1.N < e2.N)
            {
                return -1;
            } else {
                return 1;
            }
            }
        }
    }     
      private boolean AllNumericTrainingData = true; 
        //  public IntOption KValueOption = new IntOption("KValue",'k', "The number of instances nearest to use (Make odd).",3, 1, Integer.MAX_VALUE);
        //  public IntOption MaxTrainingItemsOption = new IntOption("MAXTrainingItems",'x', "The max number of instances keep.",-1, -1, Integer.MAX_VALUE);
        //  public FlagOption KMeanNoiseSupressionOption = new FlagOption("KMean_Training_of_Data",'e', "The number of instances nearest to use (Make odd).");   
        public double[] getclassIndexByMinDistance(double[][] distances)
        {           
            
            // DIST 
            // [DIST][INDEX of MC]
            
            
            double MinIndex[] = new double[2];
            //MinIndex[0]  = -1; // Inital ERROR Value.... 
            MinIndex[0]  = 0; // Inital ERROR Value.... 
            double MinVal = Double.MAX_VALUE;
            //Different OPTSD #       
            
            for (int i = 0; i < distances.length; i++) 
            {
                if (distances[i][0] < MinVal ) 
                {
                    MinVal = distances[i][0];
                    MinIndex[0] = i;
                    MinIndex[1] = distances[i][1];
                }
            }            
            
            // [ClassID][INDEXinArray]
            return MinIndex;
        }
    
      
        public void SetClassandAttributeCount(int ClassCount , int AttributeCount)
        {
            NumClasses=ClassCount;
            NumAttributes=AttributeCount; 
          
            SplitIsHappened = false;
        }
        
        
        private double[][] getAllDistances(int numClasses, double[] InstD)  //called by test and train
        {
                double[][] distances = new double[numClasses][2];
                
                for (int i = 0; i < numClasses; i++) 
                {
                    distances[i] =  getNearsetMCDistance(i,InstD);
                }
                return distances;
        }
      
        
	public double[] test(double[] InstD) 
        {
            int ClassIndex = (int)InstD[InstD.length-1];
            
            double[] votes = new double[NumClasses];    
            int WinningIndex = 0; 
            if (this.b_Initalised)
            {
                WinningIndex = (int)getclassIndexByMinDistance(getAllDistances(NumClasses,InstD))[0];
                
                if (WinningIndex < 0 ||  WinningIndex > 10)
                {
                    int n =0; 
                }
                
                
                votes[WinningIndex] = 1;
            }
            
            //if(ClassIndex == WinningIndex)
            //    CorrectPredictTest++;
            
            if(!b_Initalised){
                CorrectPredictTestArray = new int[NumClasses][NumClasses];
                for(int i=0; i<NumClasses; i++){
                    for(int j=0; j<NumClasses; j++){
                        CorrectPredictTestArray[i][j] = 0;
                    }
                }
            }
        
            CorrectPredictTestArray[ClassIndex][WinningIndex]++;
            
            return votes; 
	}
        
        //@Override
	public void resetLearningImpl(
            int ErrorOptionD, 
            int MAXClusterArraySizeOptionD,
            int RemovePoorlyPerformingClustersOptionD,
            boolean SplitOnAllAttributesFlagOptionD, 
            boolean UseGlobalTimeStampsOptionD, 
            boolean SplitClusterGetsNewTimeStampsOptionD,
            boolean DangerTheoryFlagOptionD, 
            boolean DoublePunishFlagOptionD
            ) 
        {
                b_Initalised= false; 
                
                ErrorOption = ErrorOptionD;
                MAXClusterArraySizeOption=MAXClusterArraySizeOptionD;
                RemovePoorlyPerformingClustersOption=RemovePoorlyPerformingClustersOptionD;
                SplitOnAllAttributesFlagOption=SplitOnAllAttributesFlagOptionD;
                UseGlobalTimeStampsOption=UseGlobalTimeStampsOptionD;
                SplitClusterGetsNewTimeStampsOption=SplitClusterGetsNewTimeStampsOptionD;
                DangerTheoryFlagOption=DangerTheoryFlagOptionD;
                DoublePunishFlagOption=DoublePunishFlagOptionD;
	}

        public String PrintClusters()
        {
            StringBuffer SB  =  new StringBuffer();
            SB.append("{");
             
                    for (int i = 0; i < this.Clusters.length; i++) 
                    {
                        for (int j = 0; j < this.MicroClustersCount[i] ; j++) 
                        {
                            // PRINT THE CENTERS 
                            MicroCluster MC = Clusters[i][j];
                            // SB.append("CENTERS>>");
                            for (int k = 0; k < MC.Centers.length; k++) 
                            {
                                SB.append(",");
                                SB.append("[");
                                SB.append(i);
                                SB.append("][");
                                SB.append(j);
                                SB.append("]{");
                                SB.append(k);
                                SB.append("}");
                                
                                SB.append(MC.Centers[k]);
                            } 
                        }
                       SB.append("}");
                    }
                   
                return SB.toString();
        }
        
        
          public void PrintInsertDeleteCount(String Header) throws UnsupportedEncodingException, IOException
        { 
            
            String OutPutFile = OutputPath +  "\\InsertDelete.csv";
            StringBuffer SB  =  new StringBuffer();
             
            
            try {   
                    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter( new FileOutputStream(OutPutFile, true), "UTF-8"));
                    //writer.newLine();
                    //writer.append(Header);
                    //writer.newLine();
                    
                    //for (int i = 0; i < this.Clusters.length; i++) 
                    //{
                        SB.append("Insert / Split / Delete,");
                        
                        SB.append(this.InsertCount);
                        SB.append(",");
                        SB.append(this.SplitCount);
                        SB.append(",");
                        SB.append(this.DeleteCount);
                        SB.append(",");
                        SB.append(",");
                        SB.append(",");
                        
                        int TOTAL = 0; 
                        for (int i = 0; i < this.Clusters.length; i++) 
                        {
                            int Num  = this.MicroClustersCount[i];
                            SB.append(",");    
                            SB.append(Num);
                            TOTAL = TOTAL + Num;
                        }
                        
                        SB.append(",");  
                        SB.append(TOTAL);
                        
                        writer.write(SB.toString());
                        SB = new StringBuffer();
                        writer.newLine();
                        writer.flush();
                   // }
                   
                } catch (FileNotFoundException ex) 
                {
                  //  Logger.getLogger(MicroClustersMytosis.class.getName()).log(Level.INFO, null, ex);
                }
            
            //System.out.println("Split count: "+this.SplitCount + " Death count: "+ this.DeleteCount);
            
            this.InsertCount = 0; 
            this.DeleteCount=0; 
            this.SplitCount=0; 
            
        }
           
        public void SetUPClusterArrays()
        { 
            
            this.Clusters = new MicroCluster[NumClasses][MAXClusterArraySizeOption];
            MicroClustersCount = new int[NumClasses];
            MicroClustersErrorCount = new int[NumClasses][MAXClusterArraySizeOption];
            
            StreamSquareTotal = new double[NumClasses];
            StreamItemCount = new int[NumClasses];
            StreamTotal = new double[NumClasses];
            
            trainNNIndex = new int[NumClasses]; 
            trainNNMicroCluster = new MicroCluster[NumClasses];  
            trainNNDistance = new double[NumClasses];
            
            for (int i = 0; i < NumClasses; i++) 
            {
                for (int j = 0; j < MAXClusterArraySizeOption; j++) 
                {
                    Clusters[i][j] = new MicroCluster(i,NumAttributes-1, this.StreamItemCount[i]);
                    MicroClustersErrorCount[i][j] = 0;  
                }
            }
            
            for (int i = 0; i < NumClasses; i++) 
            {
                MicroClustersCount[i] = 1; 
            }
            
            b_Initalised = true;
        }
    
        
        private int GetStreamCount(int ClassIndex)
        {
            
            if (this.UseGlobalTimeStampsOption)
            {
                return this.StreamItemCount[0];
            }
            else
            {
                return this.StreamItemCount[ClassIndex];
            }
        }
         
        public void CheckAndRemoveClusters(int ClassIndex) throws IOException
        {            
            if (this.MicroClustersCount[ClassIndex] > 1)
            {
                for (int i = 0; i < this.MicroClustersCount[ClassIndex]; i++)             
                {
                    MicroCluster MC = this.Clusters[ClassIndex][i];
                    double ClusterWeight = MC.CalculateMyBigWeight(GetStreamCount(ClassIndex));
                    
                    if (RemovePoorlyPerformingClustersOption > ClusterWeight )
                    {
                        if (this.trainingcount > 40000)
                        {
                            int nn =0 ;
                        }   
                        //double dd = MC.CalculateMyBigWeight(GetStreamCount(ClassIndex));
                        this.removeMicroCluster(ClassIndex, i);
                        //System.out.println(ClusterWeight);
                        //System.out.println(" MC Removed ");
                        
                        //PrintClusters("<<<<<< REMOVAL");
                        
                        break;
                    }
                }
            }
        }
        
        
        public void UpdateMicroClusterTimeStemaps(int ClassIndex)
        {
            for (int i = 0; i < this.MicroClustersCount[ClassIndex]; i++) 
            {
                MicroCluster MC = this.Clusters[ClassIndex][i];
                MC.IncrementTestingAges();
            }
        }
        
        
        private void IncrementStreamCount(int ClassIndex)
        {
            if (this.UseGlobalTimeStampsOption)
            {
                this.StreamItemCount[0]++;
            }
            else
            {
                this.StreamItemCount[ClassIndex]++;
                
            }
            
            
        }
       
	public void train(double[] InstD) 
        {    
         
            int ClassIndex = (int)InstD[InstD.length-1];
            
            if (!b_Initalised)
            {
                SetUPClusterArrays();
            }
              
            
            trainingcount++;
            IncrementStreamCount(ClassIndex);
            
            //remove Poorly Performing MicroCLusters
            
            //UpdateMicroClusterTimeStemaps(ClassIndex);        
          for (int i = 0; i < this.Clusters.length; i++) 
         {
             try
             {
                 
                CheckAndRemoveClusters(i);
                    //System.out.println("Tri Angle");
                
             } 
             catch (IOException ex) 
             {
                //Logger.getLogger(MicroClustersMytosis.class.getName()).log(Level.SEVERE, null, ex);
             }
         }
          
            //  NOT *****  not all Classes  need to be checked ---  as just httis class's class time incremnent has occurred!!  
            int MinClassIndex = Integer.MAX_VALUE;
            double MinClassDistnace = Double.MAX_VALUE;
            
            double[][] AllDistancesbyClass = getAllDistances(NumClasses,InstD);
                        
            for (int i = 0; i < this.Clusters.length; i++) 
            {
                //try
                 // {
                    //CheckAndRemoveClusters(i);
                    double[] ClassIndexValues  = AllDistancesbyClass[i];
                    trainNNIndex[i] = (int)ClassIndexValues[1];
                    trainNNDistance[i] = ClassIndexValues[0];
                    trainNNMicroCluster[i] =  this.Clusters[(ClassIndex)][trainNNIndex[i]];

                    if (MinClassDistnace  >= trainNNDistance[i])
                    {
                        MinClassDistnace = trainNNDistance[i];
                        MinClassIndex = i; 
                    }

                //} 
                //catch (IOException ex) 
               // {
               // Logger.getLogger(MicroClustersMytosis.class.getName()).log(Level.SEVERE, null, ex);
               //}
           }
            
            // Should we be adding this or a New Concept Cluster???? 
            trainNNMicroCluster[ClassIndex].IncrementCluster(InstD,this.GetStreamCount(ClassIndex));
            
            if (MinClassIndex != ClassIndex)
            {
                // this Class Pubnished for not Knowing this data....
                this.MicroClustersErrorCount[ClassIndex][trainNNIndex[ClassIndex]]++;
        
                if (trainNNMicroCluster[ClassIndex].N > 2)
                {
                    AttemptSplitsforMicroCluster(ClassIndex,trainNNIndex[ClassIndex]);   // split
                }else 
                {
                    if (DoublePunishFlagOption)
                    {
                        // Undo Prevoius Insert as it's not going to make a difference. 
                        this.removeMicroCluster(ClassIndex,trainNNIndex[ClassIndex]);
                        InsertNewMicroCluster(ClassIndex,InstD);
                    }
                }
            
                if (MinClassIndex < 0 || MinClassIndex > 10)
                {
                    int n =0; 
                }
                
                
                //just this Class NUM-NUTS 
                this.MicroClustersErrorCount[MinClassIndex][trainNNIndex[MinClassIndex]]++;
                AttemptSplitsforMicroCluster(MinClassIndex,trainNNIndex[MinClassIndex]);      //split 
                 
            }
            else
            {   
                
               if (DangerTheoryFlagOption)
               {
                    //reduce Error Count. 
                   if (this.MicroClustersErrorCount[ClassIndex][trainNNIndex[ClassIndex]] > 0)
                    {
                          this.MicroClustersErrorCount[ClassIndex][trainNNIndex[ClassIndex]]--;
                    }
               }
            }
        }  //end train
        
      
        public void AttemptSplitsforMicroCluster(int ClassIndex,int  MCIndex )
        {
            
             MicroCluster MC = this.Clusters[ClassIndex][MCIndex];
             
             
             if (MC.N==0)
             {
                 // PROBLEM 
                 // WTF
                 
                 int i =0; 
             }
             
             
             if (MC.N > 1)
             {
                 try
                 {
                    if (this.MicroClustersErrorCount[ClassIndex][MCIndex] > ErrorOption)
                    {
                    	
                        //this.SplitCounter++;
                        
                        SplitMicroCluster(MCIndex, ClassIndex);    //split
                    }
                 }
                            catch(Exception e)
                            {
                            }
             } 
             else
             {
                 
                 //Faster an worse. 
             //   if (this.MicroClustersCount[ClassIndex] > 1)
             //   {
             //       this.removeMicroCluster(ClassIndex, MCIndex);
             //   }
             }
          }
            
        
        public void removeMicroCluster(int ClassIndex, int MCIndex)
        {
            //[][x][][][][][][L]
            //[][L][][][][][][L]
            //[][L][][][][][]
            // if MCIndex = 2 ----- Overwrite removal object with last in array and decrement pointer ...
        	
        	
            //this.DeathCounter++;
            
            this.DeleteCount++;
            Clusters[ClassIndex][MCIndex] = Clusters[ClassIndex][this.MicroClustersCount[ClassIndex]-1];          
            MicroClustersErrorCount[ClassIndex][MCIndex] = MicroClustersErrorCount[ClassIndex][this.MicroClustersCount[ClassIndex]-1]; 
            
            this.MicroClustersCount[ClassIndex]--; 
        }
        
        public void reshuffleClusterArray(int CLassIndex)
        {
            // find Least USED 
            
            double LeastCount = Double.MAX_VALUE; 
            int IndexLeast  = 0; 
            
            for (int i = 0; i < this.MAXClusterArraySizeOption-2; i++) 
            {
                //going to Remove the Oldest... smallest (sum / N) 
                MicroCluster MC = Clusters[CLassIndex][i];
                double CheckValue  = MC.CalculateMyBigWeight(this.MicroClustersCount[CLassIndex]);
                
                if ((CheckValue )< LeastCount)
                {
                    IndexLeast = i; 
                    LeastCount = CheckValue;
                }
            }
            MicroCluster TMP = Clusters[CLassIndex][IndexLeast];
            Clusters[CLassIndex][IndexLeast] = Clusters[CLassIndex][this.MAXClusterArraySizeOption-1];
            Clusters[CLassIndex][this.MAXClusterArraySizeOption-1] = TMP;
            
            // NOW RESET the Counters 
            MicroClustersErrorCount[CLassIndex][IndexLeast] = MicroClustersErrorCount[CLassIndex][this.MAXClusterArraySizeOption-1]; 
            MicroClustersErrorCount[CLassIndex][this.MAXClusterArraySizeOption-1] = 0; 
            
        }
        
        
        public void InsertNewMicroCluster(int ClassIndex, double[] instD)
        {
            
            InsertCount++;
             if (MicroClustersCount[ClassIndex] <  MAXClusterArraySizeOption)
            {
                MicroClustersCount[ClassIndex]++;
            } else 
            {
                //need to re-shuffle array so least used is on the end             
                reshuffleClusterArray(ClassIndex);
            }
             
             
             MicroCluster MCNew = Clusters[ClassIndex][MicroClustersCount[ClassIndex]-1];
             MCNew.ResetCluster();

             MCNew.IncrementCluster(instD,this.GetStreamCount(ClassIndex));
             
             Clusters[ClassIndex][MicroClustersCount[ClassIndex]-1] = MCNew;
        }
        
        public void SplitMicroCluster(int MCIndex, int ClassIndex) throws IOException
        {
            
            int AttributeIndex  = 0;
            double AttributeValue = 0;
            
            this.SplitIsHappened = true;
            this.SplitCount++;  //split
            
            if (MicroClustersCount[ClassIndex] <  MAXClusterArraySizeOption)
            {
                MicroClustersCount[ClassIndex]++;
            } else 
            {
                //need to re-shuffle array so least used is on the end             
                reshuffleClusterArray(ClassIndex);
            }
            
            MicroCluster MCSplit = Clusters[ClassIndex][MCIndex];
            //MicroCluster MCNew = Clusters[ClassIndex][MicroClustersCount[ClassIndex]-1];
            
            MicroCluster MCNew = new MicroCluster(ClassIndex  , MCSplit.NumAttributes , 0); //It has been updated by Mark on 8/3/2016
            
            AttributeIndex  =  MCSplit.findMaxVariantArrtibute();
            AttributeValue = MCSplit.findMaxVariantArrtibureValue();

            double[] Variants = MCSplit.getVariants();
            // now rescale for N = 1
            for (int i = 0; i < MCSplit.CF1X.length; i++) 
            {
                MCSplit.CF1X[i] = MCSplit.CF1X[i] / MCSplit.N;
            }
            
            double[] instDLow = MCSplit.CF1X.clone();
            double[] instDHigh = MCSplit.CF1X.clone();
                      
            if(SplitOnAllAttributesFlagOption){
                
                int size = Variants.length;//(int)NumberSelectedFeatures;
               
                // Revers for loop which stop at number attributes - number selected attributes
                
                for(int i=0; i<size; i++){//for(int i=MCSplit.NumAttributes-1; i>=MCSplit.NumAttributes-(int)NumberSelectedFeatures;i--){
                    
                    //int IndexAttribSelected = AttributesIndexes_SelectedHighToLow[i];
                    double ValueAttribSelected = Variants[i];//VariantsValues_SelectedHighToLow[i];
                    
                    //System.out.println("IndexAttribSelected: "+IndexAttribSelected);
                    
                    instDLow[i] = instDLow[i] - (ValueAttribSelected);
                    instDHigh[i] = instDHigh[i] + (ValueAttribSelected);
                  
                
                }
                
            }else{
                instDLow[AttributeIndex] = instDLow[AttributeIndex] - (AttributeValue);
                instDHigh[AttributeIndex] = instDHigh[AttributeIndex] + (AttributeValue);
            }
            
            int OldTime = (int)(MCSplit.CF1T / MCSplit.N);
            
            MCSplit.ResetCluster();
            MCNew.ResetCluster();
            
            // NEW Time STAMP 
            if (SplitClusterGetsNewTimeStampsOption)
            {
                MCSplit.IncrementCluster(instDLow,this.GetStreamCount(ClassIndex));
                MCNew.IncrementCluster(instDHigh,this.GetStreamCount(ClassIndex));
            }
            else
            {
                MCSplit.IncrementCluster(instDLow,OldTime);
                MCNew.IncrementCluster(instDHigh,OldTime);
            }
            
           
            //MCSplit.CopyData();
            //MCNew.CopyData();
            
            Clusters[ClassIndex][MCIndex] = MCSplit;
            Clusters[ClassIndex][MicroClustersCount[ClassIndex]-1] = MCNew;
            
            
            //this.PrintClusters(">>>>>> NEW INSERT");
        }
        
        public Integer getNearsetMCIndex(int classIndex, double[] instd)
        {
            double iDist = Double.MAX_VALUE;
            int iIndex = 0; 
                        
            for (int i = 0; i < this.MicroClustersCount[classIndex]; i++) 
            {
                MicroCluster MC = Clusters[classIndex][i]; 
                
                double tmpECLDist = MC.EcludianDistancefromCentroid(instd);
                double TmpWeight = MC.CalculateMyWeight(this.GetStreamCount(classIndex)); 
                
                // Want low distance and High Weight. 
                // Therefor use division. 
                
                double TmpDist = (tmpECLDist/TmpWeight);
                
                if (TmpDist < iDist)
                {
                    iIndex = i; 
                    iDist = TmpDist;
                }
            }
            return iIndex;
        }
        
         public double[] getNearsetMCDistance(int classIndex, double[] instd)
        {   
            
            // RETURN [DIST][INDEX of MC]
            double[] iDist = new double[2];
            iDist[0] = Double.MAX_VALUE;
            int iIndex = 0; 
                        
            for (int i = 0; i < this.MicroClustersCount[classIndex]; i++) 
            {
                MicroCluster MC = Clusters[classIndex][i]; 
                
                double TmpDist = MC.EcludianDistancefromCentroid(instd);
                
                if (TmpDist < iDist[0])
                {
                    iIndex = i; 
                    iDist[0] = TmpDist;
                    iDist[1] = i;
                }
            }
            return iDist;
        }
     
    public void ResetAllMicroClusters(){
        for(int ClassIndex = 0; ClassIndex<NumClasses; ClassIndex++){
            for (int i = 0; i < this.MicroClustersCount[ClassIndex]; i++){
                MicroCluster MC = Clusters[ClassIndex][i];
                MC.ResetCluster();
            }
        }
    }
  
    public void StringBuffer_Reset(){
        StrBuffer = new StringBuffer();
    }
    
    public void MicroCluster_PrintCentersMCs(){
        System.out.println("  ");
        System.out.println(" Centers ");
        for(int ClassIndex = 0; ClassIndex<NumClasses; ClassIndex++){
            for (int i = 0; i < this.MicroClustersCount[ClassIndex]; i++){
                MicroCluster MC = Clusters[ClassIndex][i];
                for(int j = 0; j < MC.Centers.length; j++){
                    System.out.print((double)MC.Centers[j]+ " , ");
                }
                System.out.println(" ");
            }
        }
    }
    
}