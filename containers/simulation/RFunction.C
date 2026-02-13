#include "TMath.h"
#include "TF1.h"

//------------ Alexa Johnson --------------
//------------ alexa@jlab.org -------------
//----------------for DVCS ----------------
//---- Last updated: August 14, 2018   ------
//Updated by Frederic from Alexa's new code (9th Nov 2018) on the 4th Dec 2018


Double_t RFunction(Int_t RunNumber, Double_t th, Double_t dp, Double_t ph, Double_t y){


  // Run number is the the run number assigned by Coda to the corresponding root file.
  // variable: th     is THETA in TARGET COORDINATES, root branch name = "L.tr.tg_th"
  // variable: dp     is DELTA in TARGET COORDINATES, root branch name = "L.tr.tg_dp"
  // variable: ph     is PHI in TARGET COORDINATES,   root branch name = "L.tr.tg_ph"
  // variable: y      is Y in TARGET COORDINATES,     root branch name = "L.tr.tg_y"

  // THIS DOES NOT CONTAIN ANY CUTS ON Z-VERTEX!


  Int_t kinparam_one;
  Int_t kinparam_two;

  //****************************************
  //****************************************
  //**           Fall 2014 runs         ****
  //****************************************
  //****************************************

  if(RunNumber>=10553 && RunNumber<=10648){
    kinparam_one = 36;
    kinparam_two = 1;
    // ideal r-cut = 0.003
  }

  //****************************************
  //****************************************
  //****        Spring 2016 runs        ****
  //****************************************
  //****************************************


  else if(RunNumber>=12508 && RunNumber<=12661){
    kinparam_one = 48;
    kinparam_two = 1;
    // ideal r-cut = 0.003
  }
  else if((RunNumber>=13000 && RunNumber<=13015) || (RunNumber>=13183 && RunNumber<=13237)){
    kinparam_one = 48;
    kinparam_two = 2;
    //ideal r-cut = 0.003
  }
  else if(RunNumber>=12838 && RunNumber<=12992){
    kinparam_one = 48;
    kinparam_two = 3;
    //ideal r-cut = 0.006
  }
  else if((RunNumber>=13100 && RunNumber<=13162) || (RunNumber>=13279 && RunNumber<=13418)){
    kinparam_one = 48;
    kinparam_two = 4;
    //ideal r-cut = 0.0025
  }


  //****************************************
  //****************************************
  //****        Fall 2016 runs        ******
  //****************************************
  //****************************************

  else if(RunNumber>=14150 && RunNumber<=14260){
    kinparam_one = 36;
    kinparam_two = 2;
    //ideal r-cut = 0.005
  }
  else if(RunNumber>=14476 && RunNumber<=14525){
    kinparam_one = 36;
    kinparam_two = 3;
    //ideal r-cut = 0.005;
  }
  else if(RunNumber>=14528 && RunNumber<=14924){
    kinparam_one = 60;
    kinparam_two = 3;
    //ideal r-cut = 0.005;
  }
  else if((RunNumber>=14270 && RunNumber<=14423) || (RunNumber>=14961 && RunNumber<=15117)){
    kinparam_one = 60;
    kinparam_two = 1;
    //ideal r-cut = 0.005;
  }



  //**************************************************

  else {
    //20190603(start)
    //not useful for now
    // printf("**********************************\n");
    // printf("The run number you are using was not identified with a given kinematic setting.  Check that the run you are using is valid, and modify this script to include it with the approriate kinematic setting. \n");
    // printf("**********************************\n");
    //20190603(finish)

  }
  //***************************************************
  //***************************************************
  //***************************************************
  //            begin R-Function here
  //***************************************************
  //***************************************************
  //***************************************************


  Double_t r_thdp_1, r_thdp_2, r_thdp_3, r_phdp_1, r_phdp_2,r_phdp_3,r_phdp_4, r_phy_1, r_phy_2, r_phy_3, r_phy_4, r_thph_1, r_thph_2, r_thph_3, r_thph_4;
  // these are the names of functions defining initial cuts.  if a cut was deemed redundant, it is assigned a large value of 100000 to not affect the resulting R-Value.
  Double_t c1, c2, c3, c4, c5, c6, c7, c8;
  Double_t p1,p2,p3;
  Double_t d1,d2;
  Double_t R;

  //**********************************************
  //**********************************************
  //------   Rfunction for x = 36 settings   -----
  //**********************************************
  //**********************************************

  if(kinparam_one==36){
    if(kinparam_two==1){
      //-----------Cuts from the theta delta plane ----------
      r_thdp_1 =-(-13.3*dp -0.55 - th)/TMath::Sqrt(TMath::Power(13.3,2)+1);
      r_thdp_2 = (-4*dp +0.18 - th)/TMath::Sqrt(TMath::Power(4,2)+1);
      r_thdp_3 = (-0.6*dp +0.07 - th)/TMath::Sqrt(TMath::Power(0.6,2)+1);
      //----------- Cuts from the phi delta plane ----------
      r_phdp_1 = (-0.15*dp + 0.03 - ph)/TMath::Sqrt(TMath::Power(0.15,2)+1);
      r_phdp_2 =- (0.125*dp - 0.03 - ph)/TMath::Sqrt(TMath::Power(0.125,2)+1);
      r_phdp_3 = -dp + 0.05;
      r_phdp_4 = dp + 0.045;
      //---------- Cuts from the phi y plane -----------
      r_phy_1 = (-0.25*y + 0.03 - ph)/TMath::Sqrt(TMath::Power(0.25,2)+1);
      r_phy_2 = (0.5775*y + 0.042 - ph)/TMath::Sqrt(TMath::Power(0.5775,2)+1);
      r_phy_3 =- (0.538*y - 0.048 - ph)/TMath::Sqrt(TMath::Power(0.538,2)+1);
      r_phy_4 =- (-0.225*y - 0.03 - ph)/TMath::Sqrt(TMath::Power(0.225,2)+1);
      //---- Cuts from the theta phi plane
      r_thph_1 = (0.06 - th);
      r_thph_2 = -(-0.05 - th);
      r_thph_3 = 100000;
      r_thph_4 = 100000;
    }
    if(kinparam_two==2){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =(13.33*dp + 0.56 + th)/TMath::Sqrt(TMath::Power(13.33,2)+1);
      r_thdp_2 = (-0.574*dp +0.072 - th)/TMath::Sqrt(TMath::Power(0.574,2)+1);
      r_thdp_3 = (-4.76*dp +0.219 - th)/TMath::Sqrt(TMath::Power(4.76,2)+1);
      //-----------Cuts from the phi delta plane ----------------
      r_phdp_1 = (-0.1*dp + 0.032 + ph)/TMath::Sqrt(TMath::Power(0.1,2)+1);
      r_phdp_2 = 100000;
      r_phdp_3 = 10000;
      r_phdp_4 = 10000;
      //-----------Cuts from the phi y plane ----------------
      r_phy_1 = (0.22*y + 0.031 + ph)/TMath::Sqrt(TMath::Power(0.22,2)+1);
      r_phy_2 = (-0.12*y + 0.028 - ph)/TMath::Sqrt(TMath::Power(0.12,2)+1);
      r_phy_3 = 10000;
      r_phy_4 = 100000;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = th + 0.05;
      r_thph_2 = 0.055 - th;
      r_thph_3 = 10000;
      r_thph_4 = 10000;
    }
    if(kinparam_two==3){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =(13.33*dp + 0.56 + th)/TMath::Sqrt(TMath::Power(13.33,2)+1);
      r_thdp_2 = (-0.574*dp +0.072 - th)/TMath::Sqrt(TMath::Power(0.574,2)+1);
      r_thdp_3 = (-4.76*dp +0.219 - th)/TMath::Sqrt(TMath::Power(4.76,2)+1);
      //-----------Cuts from the phi delta plane ----------------
      r_phdp_1 = (-0.1*dp + 0.032 + ph)/TMath::Sqrt(TMath::Power(0.1,2)+1);
      r_phdp_2 = 100000;
      r_phdp_3 = 10000;
      r_phdp_4 = 10000;
      //-----------Cuts from the phi y plane ----------------
      r_phy_1 = (0.22*y + 0.031 + ph)/TMath::Sqrt(TMath::Power(0.22,2)+1);
      r_phy_2 = (-0.12*y + 0.028 - ph)/TMath::Sqrt(TMath::Power(0.12,2)+1);
      r_phy_3 = 10000;
      r_phy_4 = 100000;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = th + 0.05;
      r_thph_2 = 0.055 - th;
      r_thph_3 = 10000;
      r_thph_4 = 10000;
    }
  }



  //**********************************************
  //**********************************************
  //------   Rfunction for x = 60 settings   -----
  //**********************************************
  //**********************************************


  if(kinparam_one == 60){
    if(kinparam_two==1){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =(13.33*dp + 0.56 + th)/TMath::Sqrt(TMath::Power(13.33,2)+1);
      r_thdp_2 = (-0.574*dp +0.072 - th)/TMath::Sqrt(TMath::Power(0.574,2)+1);
      r_thdp_3 = (-4.76*dp +0.219 - th)/TMath::Sqrt(TMath::Power(4.76,2)+1);
      //-----------Cuts from the phi delta plane ----------------
      r_phdp_1 = (-0.1*dp + 0.032 + ph)/TMath::Sqrt(TMath::Power(0.1,2)+1);
      r_phdp_2 = 100000;
      r_phdp_3 = 10000;
      r_phdp_4 = 10000;
      //-----------Cuts from the phi y plane ----------------
      r_phy_1 = (0.22*y + 0.031 + ph)/TMath::Sqrt(TMath::Power(0.22,2)+1);
      r_phy_2 = (-0.12*y + 0.028 - ph)/TMath::Sqrt(TMath::Power(0.12,2)+1);
      r_phy_3 = 10000;
      r_phy_4 = 100000;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = th + 0.05;
      r_thph_2 = 0.055 - th;
      r_thph_3 = 10000;
      r_thph_4 = 10000;
    }
    if(kinparam_two==3){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =(13.33*dp + 0.56 + th)/TMath::Sqrt(TMath::Power(13.33,2)+1);
      r_thdp_2 = (-0.574*dp +0.072 - th)/TMath::Sqrt(TMath::Power(0.574,2)+1);
      r_thdp_3 = (-4.76*dp +0.219 - th)/TMath::Sqrt(TMath::Power(4.76,2)+1);
      //-----------Cuts from the phi delta plane ----------------
      r_phdp_1 = (-0.1*dp + 0.032 + ph)/TMath::Sqrt(TMath::Power(0.1,2)+1);
      r_phdp_2 = 100000;
      r_phdp_3 = 10000;
      r_phdp_4 = 10000;
      //-----------Cuts from the phi y plane ----------------
      r_phy_1 = (0.22*y + 0.031 + ph)/TMath::Sqrt(TMath::Power(0.22,2)+1);
      r_phy_2 = (-0.12*y + 0.028 - ph)/TMath::Sqrt(TMath::Power(0.12,2)+1);
      r_phy_3 = 10000;
      r_phy_4 = 100000;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = th + 0.05;
      r_thph_2 = 0.055 - th;
      r_thph_3 = 10000;
      r_thph_4 = 10000;
    }
  }


  //**********************************************
  //**********************************************
  //------   Rfunction for x = 48 settings   -----
  //**********************************************
  //**********************************************

  if(kinparam_one == 48){
    if(kinparam_two==1){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =-(-13.3*dp -0.55 - th)/TMath::Sqrt(TMath::Power(13.3,2)+1);
      r_thdp_2 = (-4*dp +0.18 - th)/TMath::Sqrt(TMath::Power(4,2)+1);
      r_thdp_3 = (-0.6*dp +0.07 - th)/TMath::Sqrt(TMath::Power(0.6,2)+1);
      //----------- Cuts from the phi delta plane ---------------
      r_phdp_1 = (-0.2*dp + 0.021 - ph)/TMath::Sqrt(TMath::Power(0.2,2)+1);
      r_phdp_2 =- (0.125*dp - 0.03 - ph)/TMath::Sqrt(TMath::Power(0.125,2)+1);
      r_phdp_3 = -dp + 0.043;
      r_phdp_4 = dp + 0.04;
      //---------- Cuts from the phi y plane -----------
      r_phy_1 = (-0.27*y + 0.026 - ph)/TMath::Sqrt(TMath::Power(0.27,2)+1);
      r_phy_2 = (0.5775*y + 0.042 - ph)/TMath::Sqrt(TMath::Power(0.5775,2)+1);
      r_phy_3 =- (0.538*y - 0.048 - ph)/TMath::Sqrt(TMath::Power(0.538,2)+1);
      r_phy_4 =- (-0.225*y - 0.03 - ph)/TMath::Sqrt(TMath::Power(0.225,2)+1);
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = (0.06 - th);
      r_thph_2 = -(-0.05 - th);
      r_thph_3 = 0.028 + ph;
      r_thph_4 = 0.02 - ph;
    }
    if(kinparam_two==2){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =-(-4*dp -0.17 - th)/TMath::Sqrt(TMath::Power(4,2)+1);
      r_thdp_2 = 0.035 - dp;
      r_thdp_3 = (-0.4*dp +0.04 - th)/TMath::Sqrt(TMath::Power(0.4,2)+1);
      //----------- Cuts from the phi delta plane ---------------
      r_phdp_1 =(-2.5*dp + 0.0825 - ph)/TMath::Sqrt(TMath::Power(2.5,2)+1);
      r_phdp_2 =- (0.1*dp - 0.042 - ph)/TMath::Sqrt(TMath::Power(0.1,2)+1);
      r_phdp_3 = (-0.148*dp + 0.025 - ph)/TMath::Sqrt(TMath::Power(0.148,2)+1);
      r_phdp_4 = (6.56*dp + 0.24 - ph)/TMath::Sqrt(TMath::Power(6.56,2)+1);
      //---------- Cuts from the phi y plane -----------
      r_phy_1 = (-0.075*y + 0.0285 - ph)/TMath::Sqrt(TMath::Power(0.075,2)+1);
      r_phy_4 =- (-0.325*y - 0.04 - ph)/TMath::Sqrt(TMath::Power(0.325,2)+1);
      r_phy_2 = 100000;
      r_phy_3 = 100000;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = (0.0325 - th);
      r_thph_2 = -(-0.025 - th);
      r_thph_3 = 0.038 + ph;
      r_thph_4 = 0.015 - ph;
    }
    if(kinparam_two==3){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_2 = (-2.96*dp +0.137 - th)/TMath::Sqrt(TMath::Power(2.96,2)+1);
      r_thdp_3 = (-0.492*dp +0.054 - th)/TMath::Sqrt(TMath::Power(0.492,2)+1);
      r_thdp_1 =-(-7.14*dp -0.293 - th)/TMath::Sqrt(TMath::Power(7.14,2)+1);
      //----------- Cuts from the phi delta plane ---------------
      r_phdp_1 =  (-4.2*dp + 0.146 - ph)/TMath::Sqrt(TMath::Power(4.2,2)+1);
      r_phdp_2 =- (0.1075*dp - 0.038 - ph)/TMath::Sqrt(TMath::Power(0.1075,2)+1);
      r_phdp_3 =  (15.67*dp + 0.521 - ph)/TMath::Sqrt(TMath::Power(15.67,2)+1);
      r_phdp_4 = 100000;
      //---------- Cuts from the phi y plane -----------
      r_phy_1 =(-0.225*y + 0.03 - ph)/TMath::Sqrt(TMath::Power(0.225,2)+1);
      r_phy_2 =- (-0.325*y - 0.038 - ph)/TMath::Sqrt(TMath::Power(0.325,2)+1);
      r_phy_3 = 10000;
      r_phy_4 = 10000;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_1 = (0.05 - th);
      r_thph_2 = -(-0.036 - th);
      r_thph_3 = 0.037 + ph;
      r_thph_4 = 0.02 - ph;
    }
    if(kinparam_two==4){
      //-----------Cuts from the theta delta plane ----------------
      r_thdp_1 =-(-4.5*dp -0.19 - th)/TMath::Sqrt(TMath::Power(4.5,2)+1);
      r_thdp_2 = 0.035 - dp;
      r_thdp_3 = (-0.45*dp +0.045 - th)/TMath::Sqrt(TMath::Power(0.45,2)+1);
      //----------- Cuts from the phi delta plane ---------------
      r_phdp_1 = (-0.27*dp + 0.015 - ph)/TMath::Sqrt(TMath::Power(0.27,2)+1);
      r_phdp_2 =- (0.1*dp - 0.035 - ph)/TMath::Sqrt(TMath::Power(0.1,2)+1);
      r_phdp_3 = dp + 0.03;
      r_phdp_4 = (-1.4*dp + 0.038 - ph)/TMath::Sqrt(TMath::Power(1.4,2)+1);
      //---------- Cuts from the phi y plane -----------
      r_phy_1 = -(-0.217*y - 0.032 - ph)/TMath::Sqrt(TMath::Power(0.217,2)+1);
      r_phy_2 = 100000;
      r_phy_3 = 100000;
      r_phy_4 = -ph + 0.02;
      //-----------Cuts from the theta phi plane ----------------
      r_thph_2 = -(-0.03 - th);
      r_thph_1 = (0.038 - th);
      r_thph_3 = 0.01 - ph;
      r_thph_4 = 0.03 + ph;
    }
  }

  //**********************************************
  //**********************************************
  //------   Find min value = R value        -----

  c1 = TMath::Min(r_thdp_1,r_thdp_2);
  c2 = TMath::Min(r_thdp_3,r_thph_1);
  c3 = TMath::Min(r_phdp_1,r_phdp_2);
  c4 = TMath::Min(r_phy_1,r_phy_2);
  c5 = TMath::Min(r_phy_3,r_phy_4);
  c6 = TMath::Min(r_phdp_3,r_thph_2);
  c6 = TMath::Min(c6,r_thph_3);
  c6 = TMath::Min(c6,r_thph_4);
  c6 = TMath::Min(r_phdp_4,c6);

  //----------
  p1 = TMath::Min(c1,c2);
  p2 = TMath::Min(c3,c4);
  p3 = TMath::Min(c5,c6);

  d1 = TMath::Min(p1,p2);
  d2 = TMath::Min(d1,p3);

  R = TMath::Min(d1,d2);

  return R;
}
