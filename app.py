import streamlit as st
import pickle 
import numpy as np
import time

##############################################################
######################## LOAD MODEL ##########################
##############################################################
gbc_model = pickle.load(open(r"gbc_model.sav", "rb"))


##############################################################
#################### PREDICTION FUNCTION #####################
##############################################################
def mental_health_prediction(feeling_nervous, feeling_panic, breathing_rapidly, sweating, trouble_concentration, trouble_sleeping, trouble_with_work, hopelessness, anger, overreact, change_in_eating, suicidal_thought, feeling_tired, close_friend, social_media_addiction, weight_gain, material_possessions, introvert, popping_up_stressful_memory, having_nightmares, avoid_people_or_activities, feeling_negative, blaming_yourself): 
        # change the following attributes into numerical features
        if feeling_nervous == 'No':
                feeling_nervous = 0
        else:
                feeling_nervous = 1

        if feeling_panic == 'No':
                feeling_panic = 0
        else:
                feeling_panic = 1
        
        if breathing_rapidly == 'No':
                breathing_rapidly = 0
        else:
                breathing_rapidly = 1

        if sweating == 'No':
                sweating = 0
        else:
                sweating = 1
        
        if trouble_concentration == 'No':
                trouble_concentration = 0
        else:
                trouble_concentration = 1
        
        if trouble_sleeping == 'No':
                trouble_sleeping = 0
        else:
                trouble_sleeping = 1

        if trouble_with_work == 'No':
                trouble_with_work = 0
        else:
                trouble_with_work = 1
        
        if hopelessness == 'No':
                hopelessness = 0
        else:
                hopelessness = 1
        
        if anger == 'No':
                anger = 0
        else:
                anger = 1

        if overreact == 'No':
                overreact = 0
        else:
                overreact = 1
        
        if change_in_eating == 'No':
                change_in_eating = 0
        else:
                change_in_eating = 1

        if suicidal_thought == 'No':
                suicidal_thought = 0
        else:
                suicidal_thought = 1

        if feeling_tired == 'No':
                feeling_tired = 0
        else:
                feeling_tired = 1
        
        if close_friend == 'No':
                close_friend = 0
        else:
                close_friend = 1
        
        if social_media_addiction == 'No':
                social_media_addiction = 0
        else:
                social_media_addiction = 1

        if weight_gain == 'No':
                weight_gain = 0
        else:
                weight_gain = 1
        
        if material_possessions == 'No':
                material_possessions = 0
        else:
                material_possessions = 1
        
        if introvert == 'No':
                introvert = 0
        else:
                introvert = 1

        if popping_up_stressful_memory == 'No':
                popping_up_stressful_memory = 0
        else:
                popping_up_stressful_memory = 1
        
        if having_nightmares == 'No':
                having_nightmares = 0
        else:
                having_nightmares = 1
        
        if avoid_people_or_activities == 'No':
                avoid_people_or_activities = 0
        else:
                avoid_people_or_activities = 1

        if feeling_negative == 'No':
                feeling_negative = 0
        else:
                feeling_negative = 1
        
        if blaming_yourself == 'No':
                blaming_yourself = 0
        else:
                blaming_yourself = 1
        
        # combine the transformed input data into an array list
        input_data = np.array([feeling_nervous, feeling_panic, breathing_rapidly, sweating, trouble_concentration, trouble_sleeping, trouble_with_work, hopelessness, anger, overreact, change_in_eating, suicidal_thought, feeling_tired, close_friend, social_media_addiction, weight_gain, material_possessions, introvert, popping_up_stressful_memory, having_nightmares, avoid_people_or_activities, feeling_negative, blaming_yourself]).reshape(1, -1)

        # predict the mental health disorder using the trained gradient boosting model 
        predicted_proba = gbc_model.predict_proba(input_data)
        predicted_class = gbc_model.predict(input_data)
        return predicted_proba, predicted_class


##############################################################
################### SIDEBAR SECTION ##########################
##############################################################
st.sidebar.image("images/mental_words.png")
st.sidebar.header("Synopsis")
st.sidebar.write("""
                 Mental illness is a health problem that undoubtedly impacts emotions, reasoning, and social interaction of a person.
                 These issues have shown that mental illness gives serious repercussions across societies and demands new strategies for prevention and intervention. 
                 To accomplish these strategies, early detection of mental health is an imperative procedure and usually diangosed based on individual questionnaires report for detection of the specific ptterns of feeling or social interactions.
                 """)
                 
st.sidebar.header("What's so unique about this health & wellness app?")
st.sidebar.write("""
                 This health & wellness app has been integrated with an optimized ensemble machine learning algorithm - **Gradient Boosting Classifier**. Gradient Boosting Classifier
                 is a supervised learning ML that is widely applied approach in many types of research, studies, and experiments, especially in predicting illness in the medical field. In this case, the supervised learning is a classification technique 
                 using structured training data that consists of the attributes and target classes. 
                 
                 Gradient Boosting Classifier has been trained using 80% of the collected dataset and optimized it with hyperparameters such as 
                 - the number of trees in the forest (i.e. n_estimators), 
                 - how deep the built tree should be (i.e. max_depth), and 
                 - how fast the model should learn (i.e. learning_rate).
                 
                 """)
st.sidebar.image("images/model_training.jpg")

st.sidebar.header("How accurate, sensitive, and precise of the Gradient Boosting Classifier is?")
st.sidebar.image("images/question.jpg", width = 150)
st.sidebar.write("""
                 The remaining 20% of the collected dataset has been used to evaluate the reliability and predictability of the training model (i.e. Gradient Boosting Classifier).
                 The evaluation outcomes of the proposed Gradient Boosting Classifier are shown below: 
                 """)
st.sidebar.image("images/class_report_train.jpg")
st.sidebar.image("images/class_report_test.jpg")



#############################################################
################### HEADER SECTION ##########################
#############################################################
st.title("Mental Health & Wellness")
st.image("images/mental_wellness.png")
st.warning("""**DISCLAIMER:** This health & wellness app is meant for **academic-purpose** and has not been approved by the clinical practitioner global. As such, 
        this app cannot be used to replace clinical practitioners to determine your mental health status. If you feel that your mental health condition has not been improving,
        please kindly consult your medical professional immediately. 
        """)
st.info("""
        **NOTE:** This health & wellness app does not provides prediction on wide-spectrum of mental-health disorders as it only provides predictions on anxiety, depression, loneliness, and stress. 
        """)


############################################################
################### FORM SECTION ###########################
############################################################
with st.form(key = 'form1'):
        st.info("""**NOTE:** Please kindly fill out all the **required** fields below truthfully and 
                click on "predict" button to determine your mental health status. 
        
                """)
        
        feeling_nervous = st.radio("1. Are you feeling nervous most of the time?", ['Yes', 'No'])
        feeling_panic = st.radio("2. Do you have a panic attack episode recently? ", ['Yes', 'No'])
        breathing_rapidly = st.radio("3. Do you experience rapid breathing when it comes to social situation? ", ['Yes', 'No'])
        sweating = st.radio("4. Do you experience excessive sweating when it comes to social situation? ", ['Yes', 'No'])
        trouble_concentration = st.radio("5. Do you have trouble with concentration coping in daily life?", ['Yes', 'No'])
        trouble_sleeping = st.radio("6. Do you have trouble with sleeping?", ['Yes', 'No'])
        trouble_with_work = st.radio("7. Do you have trouble with work? ", ['Yes', 'No'])
        hopelessness = st.radio("8. Do you feel hopelessness in life?", ['Yes', 'No'])
        anger = st.radio("9. Do you feel anger for no other reason? ", ['Yes', 'No'])
        overreact = st.radio("10. Are you over-reacting? ", ['Yes', 'No'])
        change_in_eating = st.radio("11. Did you experience any changes in eating? ", ['Yes', 'No'])
        suicidal_thought = st.radio("12. Have you ever thought of commit suicide? ", ['Yes', 'No'])
        feeling_tired = st.radio("13. Are you feeling tired? ", ['Yes', 'No'])
        close_friend = st.radio("14. Do you have any close friends?", ['Yes', 'No'])
        social_media_addiction = st.radio("15. Are you addicted to social media? ", ['Yes', 'No'])
        weight_gain = st.radio("16. Do you feel that you are gaining weight? ", ['Yes', 'No'])
        material_possessions = st.radio("17. Are you materialistic when it comes to treasured possessions? ", ['Yes', 'No'])
        introvert = st.radio("18. Are you an introvert? ", ['Yes', 'No'])
        popping_up_stressful_memory = st.radio("19. Are you experiencing popping up stressful memory? ", ['Yes', 'No'])
        having_nightmares = st.radio("20. Do you experience any nightmares? ", ['Yes', 'No'])
        avoid_people_or_activities = st.radio("21. Do you withdraw from social life? ", ['Yes', 'No'])
        feeling_negative = st.radio("22. Do you have any negative thoughts?", ['Yes', 'No'])
        blaming_yourself = st.radio("23. Do you often blaming yourself? ", ['Yes', 'No'])

        # prompt the user to predict mental health disorder using the submit buttom
        submit_button = st.form_submit_button("Predict")

        if submit_button:
                # call the function to predict the type of mental health disorder and its probability
                predicted_proba_, predicted_class_ = mental_health_prediction(feeling_nervous,
                                                                        feeling_panic,
                                                                        breathing_rapidly,
                                                                        sweating,
                                                                        trouble_concentration,
                                                                        trouble_sleeping,
                                                                        trouble_with_work,
                                                                        hopelessness,
                                                                        anger,
                                                                        overreact,
                                                                        change_in_eating,
                                                                        suicidal_thought,
                                                                        feeling_tired,
                                                                        close_friend,
                                                                        social_media_addiction,
                                                                        weight_gain,
                                                                        material_possessions,
                                                                        introvert,
                                                                        popping_up_stressful_memory,
                                                                        having_nightmares,
                                                                        avoid_people_or_activities,
                                                                        feeling_negative,
                                                                        blaming_yourself)
                
                with st.spinner("In progress..."):
                    time.sleep(6)
                
                
                if predicted_class_ == 0:
                        proba = predicted_proba_[:,0]*100
                        st.error("**Result:** You might have about a {:.2f}% chance of suffering from anxiety disorder.".format(float(proba)))
                        st.info("""
                                Here are some of the management options for anxiety disorders:
                                - Learning about anxiety
                                - Relaxation techniques
                                - Correct breathing techniques
                                - Dietary adjustments
                                - Exercise regularly
                                - Building self-esteem
                                - Structured problem-solving
                                - Medication
                                - Get enough sleep
                                - Forming support groups
                                """)
                        st.write("You may visit this link to know more about anxiety disorder: https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/anxiety-disorders")
                elif predicted_class_ == 1:
                        proba = predicted_proba_[:,1]*100
                        st.error("**Result:** You might have about a {:.2f}% chance of suffering from a depressive disorder.".format(float(proba)))
                        st.info("""
                                Here are some of the management options for depressive disorders:
                                - Learning about depressive
                                - Lifestyle changes such as dietary adjustment and exercise regularly
                                - Attend psychological therapy provided by a mental health professional or via online e-therpaies
                                """)
                        st.write("You may visit this link to know more about depressive disorders: https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/depression-treatment-and-management")
               
                elif predicted_class_ == 2:
                        proba = predicted_proba_[:,2]*100
                        st.error("**Result:** You might have about a {:.2f}% chance of suffering from loneliness.".format(float(proba)))
                        st.info("""
                                Here are some of the management options for loneliness:
                                - Make new connections
                                - Try to open up
                                - Talking therapies
                                - Be careful when comparing yourself with others
                                - Look after yourself
                                - Take it slow
                                """)
                        st.write("You may visit this link to know more about loneliness: https://www.mind.org.uk/information-support/tips-for-everyday-living/loneliness/tips-to-manage-loneliness/")
                elif predicted_class_ == 3:
                        proba = predicted_proba_[:,3]*100
                        st.success("**Result:** You have about a {:.2f}% chance of not suffering from any mental health disorder.".format(float(proba)))
                        st.info("""
                                Here are some of the management options to look after your healthy minds:
                                - Relaxation techniques
                                - Dietary adjustments
                                - Exercise regularly
                                - Get enough sleep
                                - Forming support groups
                                """)
                else:
                        proba = predicted_proba_[:,4]*100
                        st.error("**Result:** You might have about a {:.2f}% chance of suffering from stress. ".format(float(proba)))
                        st.info("""
                                Here are some of the management options for stress:
                                - Learning about the cause of stress
                                - Relaxation techniques
                                - Dietary adjustments
                                - Exercise regularly
                                - Get enough sleep
                                - Forming support groups
                                """)
                        st.write("You may visit this link to know more about stress: https://www.betterhealth.vic.gov.au/health/healthyliving/stress")
