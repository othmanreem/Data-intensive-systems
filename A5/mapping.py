FMS_mapping_numbers = {
    "upper_body": {
    "1": ["SpineBase", "-Mid", "-Shoulder"],
    "2": ["SpineMid", "-Shoulder", "Neck"],
    "3": ["SpineShoulder", "Neck", "Head"],
    "4": ["SpineShoulder", "ShoulderLeft", "ElbowLeft"],
    "5": ["ShoulderLeft", "ElbowLeft", "WristLeft"],
    "6": ["SpineShoulder", "ShoulderRight", "ElbowRight"],
    "7": ["ShoulderRight", "ElbowRight", "WristRight"]
    },
    "lower_body": {
    "8": ["SpineMid", "-Base", "HipLeft"],
    "9": ["SpineBase", "HipLeft", "KneeLeft"],
    "10": ["HipLeft", "KneeLeft", "AnkleLeft"],
    "11": ["SpineMid", "-Base", "HipRight"],
    "12": ["SpineBase", "HipRight", "KneeRight"],
    "13": ["HipRight", "KneeRight", "AnkleRight"]
    }
}

FMS_mapping = {
"upper_body": {
"No_1_Angle_Deviation": ["SpineBase", "-Mid", "-Shoulder"],
"No_2_Angle_Deviation": ["SpineMid", "-Shoulder", "Neck"],
"No_3_Angle_Deviation": ["SpineShoulder", "Neck", "Head"],
"No_4_Angle_Deviation": ["SpineShoulder", "ShoulderLeft", "ElbowLeft"],
"No_5_Angle_Deviation": ["ShoulderLeft", "ElbowLeft", "WristLeft"],
"No_6_Angle_Deviation": ["SpineShoulder", "ShoulderRight", "ElbowRight"],
"No_7_Angle_Deviation": ["ShoulderRight", "ElbowRight", "WristRight"]
},
"lower_body": {
"No_8_Angle_Deviation": ["SpineMid", "-Base", "HipLeft"],
"No_9_Angle_Deviation": ["SpineBase", "HipLeft", "KneeLeft"],
"No_10_Angle_Deviation": ["HipLeft", "KneeLeft", "AnkleLeft"],
"No_11_Angle_Deviation": ["SpineMid", "-Base", "HipRight"],
"No_12_Angle_Deviation": ["SpineBase", "HipRight", "KneeRight"],
"No_13_Angle_Deviation": ["HipRight", "KneeRight", "AnkleRight"]
}
}

NASM_mapping = {
  "upper_body": {
    "No_14_NASM_Deviation": {
      "label": "5 FMS",
      "joints": ["ShoulderLeft", "ElbowLeft", "WristLeft"]
    },
    "No_15_NASM_Deviation": {
      "label": "7 FMS",
      "joints": ["ShoulderRight", "ElbowRight", "WristRight"]
    },
    "No_16_NASM_Deviation": {
      "label": "1 FMS",
      "joints": ["SpineBase", "SpineMid", "SpineShoulder"]
    },
    "No_19_NASM_Deviation": {
      "joints": ["ShoulderLeft", "ShoulderRight", "x-axis"]
    },
    "No_20_NASM_Deviation": {
      "joints": ["Neck", "Head", "z-axis"]
    },
    "No_21_NASM_Deviation": {
      "joints": ["ShoulderLeft", "WristLeft", "z-axis"]
    },
    "No_22_NASM_Deviation": {
      "joints": ["ShoulderRight", "WristRight", "z-axis"]
    },
    "No_23_NASM_Deviation": {
      "joints": ["SpineBase", "SpineShoulder", "z-axis"]
    },
    "No_31_NASM_Deviation": {
      "joints": ["ShoulderLeft", "WristLeft", "z-axis"]
    },
    "No_32_NASM_Deviation": {
      "joints": ["ShoulderRight", "WristRight", "z-axis"]
    },
    "No_33_NASM_Deviation": {
      "joints": ["WristLeft", "WristRight", "x-axis"]
    },
    "No_34_NASM_Deviation": {
      "joints": ["ShoulderLeft", "ShoulderRight", "x-axis"]
    },
    "No_38_NASM_Deviation": {
      "description": "Arms symmetric"
    }
  },
  "lower_body": {
    "No_17_NASM_Deviation": {
      "label": "10 FMS",
      "joints": ["HipLeft", "KneeLeft", "AnkleLeft"]
    },
    "No_18_NASM_Deviation": {
      "label": "13 FMS",
      "joints": ["HipRight", "KneeRight", "AnkleRight"]
    },
    "No_24_NASM_Deviation": {
      "joints": ["KneeLeft", "AnkleLeft", "z-axis"]
    },
    "No_25_NASM_Deviation": {
      "joints": ["KneeRight", "AnkleRight", "z-axis"]
    },
    "No_26_NASM_Deviation": {
      "joints": ["HipLeft", "KneeLeft", "y-axis"]
    },
    "No_27_NASM_Deviation": {
      "joints": ["HipRight", "KneeRight", "y-axis"]
    },
    "No_28_NASM_Deviation": {
      "joints": ["HipLeft", "AnkleLeft", "z-axis"]
    },
    "No_29_NASM_Deviation": {
      "joints": ["HipRight", "AnkleRight", "z-axis"]
    },
    "No_35_NASM_Deviation": {
      "joints": ["HipLeft", "HipRight", "x-axis"]
    },
    "No_36_NASM_Deviation": {
      "joints": ["KneeLeft", "KneeRight", "x-axis"]
    },
    "No_37_NASM_Deviation": {
      "joints": ["AnkleLeft", "AnkleRight", "x-axis"]
    }
  }
}

NASM_mapping_numbers = {
"upper_body": {
    "14": {
    "label": "5 FMS",
    "joints": ["ShoulderLeft", "ElbowLeft", "WristLeft"]
    },
    "15": {
    "label": "7 FMS",
    "joints": ["ShoulderRight", "ElbowRight", "WristRight"]
    },
    "16": {
    "label": "1 FMS",
    "joints": ["SpineBase", "SpineMid", "SpineShoulder"]
    },
    "19": {
    "joints": ["ShoulderLeft", "ShoulderRight", "x-axis"]
    },
    "20": {
    "joints": ["Neck", "Head", "z-axis"]
    },
    "21": {
    "joints": ["ShoulderLeft", "WristLeft", "z-axis"]
    },
    "22": {
    "joints": ["ShoulderRight", "WristRight", "z-axis"]
    },
    "23": {
    "joints": ["SpineBase", "SpineShoulder", "z-axis"]
    },
    "31": {
    "joints": ["ShoulderLeft", "WristLeft", "z-axis"]
    },
    "32": {
    "joints": ["ShoulderRight", "WristRight", "z-axis"]
    },
    "33": {
    "joints": ["WristLeft", "WristRight", "x-axis"]
    },
    "34": {
    "joints": ["ShoulderLeft", "ShoulderRight", "x-axis"]
    },
    "38": {
    "description": "Arms symmetric"
    }
    },
    "lower_body": {
    "17": {
    "label": "10 FMS",
    "joints": ["HipLeft", "KneeLeft", "AnkleLeft"]
    },
    "18": {
    "label": "13 FMS",
    "joints": ["HipRight", "KneeRight", "AnkleRight"]
    },
    "24": {
    "joints": ["KneeLeft", "AnkleLeft", "z-axis"]
    },
    "25": {
    "joints": ["KneeRight", "AnkleRight", "z-axis"]
    },
    "26": {
    "joints": ["HipLeft", "KneeLeft", "y-axis"]
    },
    "27": {
    "joints": ["HipRight", "KneeRight", "y-axis"]
    },
    "28": {
    "joints": ["HipLeft", "AnkleLeft", "z-axis"]
    },
    "29": {
    "joints": ["HipRight", "AnkleRight", "z-axis"]
    },
    "35": {
    "joints": ["HipLeft", "HipRight", "x-axis"]
    },
    "36": {
    "joints": ["KneeLeft", "KneeRight", "x-axis"]
    },
    "37": {
    "joints": ["AnkleLeft", "AnkleRight", "x-axis"]
    }
    }
}

weaklinkg_mapping = {
  "upper_body": [
    "ForwardHead",
    "LeftArmFallForward",
    "LeftShoulderElevation",
    "RightArmFallForward",
    "RightShoulderElevation",
    "ExcessiveForwardLean"
  ],
  "lower_body": [
    "LeftAsymmetricalWeightShift",
    "LeftHeelRises",
    "LeftKneeMovesInward",
    "LeftKneeMovesOutward",
    "RightAsymmetricalWeightShift",
    "RightHeelRises",
    "RightKneeMovesInward",
    "RightKneeMovesOutward"
  ]
}
