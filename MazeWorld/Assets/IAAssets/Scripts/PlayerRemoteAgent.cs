using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Net;
using System.Text;
using UnityStandardAssets.Characters.ThirdPerson;
using UnityEngine.SceneManagement;
using ai4u;

namespace ai4u.ext
{

    public class PlayerRemoteAgent : RLAgent
    {
        //BEGIN::Game controller variables
        private ThirdPersonCharacter character;
        private Transform m_CamTransform;
        private Vector3 m_CamForward;             // The current forward direction of the camera
        private Vector3 m_Move;
        //END::

        //BEGIN::motor controll variables
        private float fx, fy;
        private float speed = 0.0f;
        private bool crouch;
        private bool jump;
        private float leftTurn = 0;
        private float rightTurn = 0;
        private float up = 0;
        private float down = 0;
        private bool pushing;
        private bool getpickup;
        private bool usewalkspeed = false;
        private float walkspeed = 0.5f;

        public int nMoveUpdates = 4;

        public int GameID;


        public Text hud;

        public Camera m_camera;

        private Rigidbody mRigidBody;

        public float initialEnergy = 30;

        private float energy;

        public float energyRatio = 5.0f/60.0f;

        private bool done;

        public GameObject TopLeftCorner, BottonRightCorner;

        public GameObject[] respawnPositions;

        public GameObject[] fruits;
        public GameObject[] fires;

        public bool get_result = false;

        private float touchX = 0, touchY = 0, touchZ = 0;

        public bool useRayCasting = false;

        private bool isToRespawn;

        private PlayerRemoteSensor sensor;

        // Use this for initialization
        void Start()
        {
            
            fires = new GameObject[10];
            fruits = new GameObject[10];
            for (int i = 1; i <= 10; i++){
                fires[i-1] = GameObject.Find("Game" + GameID + "/FireBall" + i);
                fruits[i-1] = GameObject.Find("Game" + GameID + "/LifeBall" + i);
            }

            mRigidBody = GetComponent<Rigidbody>();
            Respawn(false);
            if (!gameObject.activeSelf)
            {
                return;
            }

            if (m_camera != null)
            {
                m_CamTransform = m_camera.transform;
            }
            else
            {
                Debug.LogWarning(
                    "Warning: no main camera found. Third person character needs a Camera tagged \"MainCamera\", for camera-relative controls.", gameObject);
                // we use self-relative controls in this case, which probably isn't what the user wants, but hey, we warned them!
            }

            // get the third person character ( this should never be null due to require component )
            character = GetComponent<ThirdPersonCharacter>();
            if (useRayCasting) {
                sensor = new PlayerRemoteSensor();
                sensor.Start(this.m_camera, gameObject, 20, 20);
            }
        }
        
        private float deltaTime = 0;

        private void Respawn(bool respawn=true){
            ResetReward();
            NotifyReset();
            if (respawn) {
                foreach(GameObject fruit in fruits){
                    fruit.GetComponent<RandomAgent>().Respawn();
                }

                foreach(GameObject fire  in fires){
                    fire.GetComponent<RandomAgent>().Respawn();
                }
            }
            getpickup = false;
            deltaTime = 0;
            energy = initialEnergy;
            ResetState();
            done = false;
            int idx = Random.Range(0, respawnPositions.Length);
            //mRigidBody.position = respawnPositions[idx].transform.position;

            Vector3 pos = respawnPositions[idx].transform.position;
            mRigidBody.velocity = Vector3.zero;
            //mRigidBody.MovePosition(pos);
            transform.position = pos;
            isToRespawn = false;
        }

        private void ResetState()
        {
            fx = 0;
            fy = 0;
            crouch = false;
            jump = false;
            pushing = false;
            leftTurn = 0;
            rightTurn = 0;
            up = 0;
            down = 0;
            get_result = false;
            ResetReward();
        }


        private void UpdateHUD(){
            if (hud != null) {
                hud.text = "Energy: " + System.Math.Round(energy,2) + "\tReward: " + reward + "\tDone: " + done + "\tGameID: " + GameID;
            }
        }

        public override void ApplyAction()
        {
            string action = GetActionName();
            if (action.Equals("SetNMoves")){
                nMoveUpdates = GetActionArgAsInt();
            } else if (action.Equals("get_result")){
                    get_result = true;
            } else if (action.Equals("restart")) {
                isToRespawn = true;
            } else if (!done) {
                switch (action)
                {
                    case "walk":
                        fx = 0;
                        fy = GetActionArgAsFloat();
                        increaseEnergy(-0.0001f);
                        break;
                    case "run":
                        fx = 0;
                        fy = GetActionArgAsFloat();
                        increaseEnergy(-0.0005f);
                        break;
                    case "walk_in_circle":
                        fx = GetActionArgAsFloat();;
                        fy = 0;
                        increaseEnergy(-0.0001f);
                        break;
                    case "right_turn":
                        rightTurn = GetActionArgAsFloat();
                        break;
                    case "left_turn":
                        leftTurn = GetActionArgAsFloat();
                        break;
                    case "up":
                        up = GetActionArgAsFloat();
                        break;
                    case "down":
                        down = GetActionArgAsFloat();
                        break;
                    case "push":
                        pushing = GetActionArgAsBool();
                        increaseEnergy(-0.01f);
                        break;
                    case "jump":
                        jump = GetActionArgAsBool();
                        increaseEnergy(-0.1f);
                        break;
                    case "crouch":
                        crouch = GetActionArgAsBool();
                        break;
                    case "pickup":
                        getpickup = GetActionArgAsBool();
                        break;
                }
            }
        }


        // Update is called once per frame
        public override void UpdatePhysics()
        {
            if (done) {
                return;
            }

            // read inputs
            float h = fx;
            float v = fy;


            // calculate move direction to pass to character
            if (m_CamTransform != null)
            {

                // calculate camera relative direction to move:
                m_CamForward = Vector3.Scale(m_CamTransform.forward, new Vector3(1, 0, 1)).normalized;
                m_Move = v * m_CamForward + h * m_CamTransform.right;

            }
            else
            {
                // we use world-relative directions in the case of no main camera
                m_Move = v * Vector3.forward + h * Vector3.right;
            }


            // walk speed multiplier
            if (usewalkspeed) {
                m_Move *= walkspeed;
            } 

            // pass all parameters to the character control script
            for (int m = 0; m < nMoveUpdates; m++) {
                character.Move(m_Move, crouch, jump, rightTurn - leftTurn, down - up, pushing, fx, fy, getpickup);
            }
            //character.Move(m_Move, crouch, m_Jump, h, v, pushing);
            jump = false;
            float x = transform.localPosition.x;
            float z = transform.localPosition.z;
            float tx = TopLeftCorner.transform.localPosition.x;
            float bx = BottonRightCorner.transform.localPosition.x;
            float tz = TopLeftCorner.transform.localPosition.z;
            float bz = BottonRightCorner.transform.localPosition.z;
            if (!done && (x < tx || x > bx || z > tz || z < bz)) {
                done = true;
                reward += 100;
            }
        }

        private void increaseEnergy(float inc){
            energy += inc;
            if (energy>50) {
                energy = 50;
            }

            if (energy < 0) {
                energy = 0;
            }
        }

        public override void touchListener(TouchRewardFunc fun){
            if (fun.gameObject.tag == "Fire") {
                increaseEnergy(-20);
            } else {
                increaseEnergy(10);
            }
            fun.gameObject.SetActive(false);
        }

        private Ray left = new Ray(), right = new Ray(), forward = new Ray(), backward = new Ray();

        private float[] GetFeaturesArray() {
            left.origin = right.origin = forward.origin = backward.origin = transform.position;
            left.direction = -1*transform.right;
            right.direction = transform.right;
            forward.direction = transform.forward;
            backward.direction = -1 * transform.forward;
            
            //4 ray + 4 agent info + 40 obj = 48 
            float[] features = new float[48];
            int f = 0;
            float N = 200.0f;
            RaycastHit hitinfo;
            if (Physics.Raycast(left, out hitinfo, 500)){
                string objname = hitinfo.collider.gameObject.name;
                features[f++] = objname == "maze1" | objname == "Terrain1" ? hitinfo.distance/N: 0;
            }
            if (Physics.Raycast(right, out hitinfo, 500)){
                string objname = hitinfo.collider.gameObject.name;
                features[f++] = objname == "maze1" | objname == "Terrain1" ? hitinfo.distance/N: 0;
            }
            if (Physics.Raycast(backward, out hitinfo, 500)){
                string objname = hitinfo.collider.gameObject.name;
                features[f++] = objname == "maze1" | objname == "Terrain1" ? hitinfo.distance/N: 0;
            }
            if (Physics.Raycast(forward, out hitinfo, 500)){
                string objname = hitinfo.collider.gameObject.name;
                features[f++] =  objname == "maze1" | objname == "Terrain1" ? hitinfo.distance/N: 0;
            }

            features[f++] = transform.position.x/N;
            features[f++] = transform.position.z/N;
            features[f++] = transform.rotation.y/360.0f;
            features[f++] = energy/30.0f;

            for (int i = 1; i <= 10; i++){
                features[f++] = fires[i-1].transform.position.x/N;
                features[f++] = fires[i-1].transform.position.y/N;
                features[f++] = fruits[i-1].transform.position.x/N;
                features[f++] = fruits[i-1].transform.position.y/N;
            }

            return features;
        }

        public override void UpdateState()
        {

            deltaTime += Time.deltaTime;
            if (deltaTime > 1.0){
                energy -= energyRatio;
                if (energy < 0){
                    energy = 0;
                    done = true;
                }
                deltaTime = 0;
            }

            if (useRayCasting) {
                byte[] frame = sensor.updateCurrentRayCastingFrame();
                SetStateAsByteArray(0, "frame", frame);
            } else {
                float[] features = GetFeaturesArray();
                SetStateAsFloatArray(0, "frame", features);
            }

            SetStateAsFloat(1, "reward", reward);
            SetStateAsFloat(2, "flag", 0);
            SetStateAsFloat(3, "energy", energy);
            SetStateAsBool(4, "done", done);
            SetStateAsFloat(5, "tx", touchX);
            SetStateAsFloat(6, "ty", touchY);
            SetStateAsFloat(7, "tz", touchZ);
            SetStateAsFloat(8, "fx", transform.forward.x);
            SetStateAsFloat(9, "fz", transform.forward.z);

            UpdateHUD();
            if(get_result) {
                ResetState();
            }

            if (isToRespawn) {
                Respawn();
            }

        }
    }

    public class PlayerRemoteSensor
    {
        private byte[] currentFrame;
        
        private Camera m_camera;

        private GameObject player;

        private int life, score;
        private float energy;


        private int verticalResolution = 20;
        private int horizontalResolution = 20;
        private bool useRaycast = true;

        private Ray[,] raysMatrix = null;
        private int[,] viewMatrix = null;
        private Vector3 fw1 = new Vector3(), fw2 = new Vector3(), fw3 = new Vector3();

        
        public void SetCurrentFrame(byte[] cf)
        {
            this.currentFrame = cf;
        }

        // Use this for initialization
        public void Start(Camera cam, GameObject player, int rayCastingHRes, int rayCastingVRes)
        {
            this.verticalResolution = rayCastingVRes;
            this.horizontalResolution = rayCastingHRes;
            life = 0;
            score = 0;
            energy = 0;
            useRaycast = true;
            currentFrame = null;

            m_camera = cam;
            this.player = player;
            fw3 = m_camera.transform.forward;


            if (useRaycast)
            {
                if (raysMatrix == null)
                {
                    raysMatrix = new Ray[verticalResolution, horizontalResolution];
                }
                if (viewMatrix == null)
                {
                    viewMatrix = new int[verticalResolution, horizontalResolution];

                }
                for (int i = 0; i < verticalResolution; i++)
                {
                    for (int j = 0; j < horizontalResolution; j++)
                    {
                        raysMatrix[i, j] = new Ray();
                    }
                }
                currentFrame = updateCurrentRayCastingFrame();
            }    
        }



        public byte[] updateCurrentRayCastingFrame()
        {
            string data = getCurrentRayCastingFrame();
            return Encoding.UTF8.GetBytes(data.ToString().ToCharArray());
        }


        public string getCurrentRayCastingFrame()
        {
            UpdateRaysMatrix(m_camera.transform.position, m_camera.transform.forward, m_camera.transform.up, m_camera.transform.right);
            UpdateViewMatrix();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < verticalResolution; i++)
            {
                for (int j = 0; j < horizontalResolution; j++)
                {
                    sb.Append(viewMatrix[i, j]);
                    if (j < horizontalResolution-1){
                        sb.Append(",");
                    }
                }
                if (i < verticalResolution-1){
                    sb.Append(";");
                }
            }
            return sb.ToString();
        }

        private void UpdateRaysMatrix(Vector3 position, Vector3 forward, Vector3 up, Vector3 right, float fieldOfView = 90.0f)
        {


            float vangle = 2 * fieldOfView / verticalResolution;
            float hangle = 2 * fieldOfView / horizontalResolution;

            float ivangle = -fieldOfView;

            for (int i = 0; i < verticalResolution; i++)
            {
                float ihangle = -fieldOfView;
                fw1 = (Quaternion.AngleAxis(ivangle + vangle * i, right) * forward).normalized;
                fw2.Set(fw1.x, fw1.y, fw1.z);

                for (int j = 0; j < horizontalResolution; j++)
                {
                    raysMatrix[i, j].origin = position;
                    raysMatrix[i, j].direction = (Quaternion.AngleAxis(ihangle + hangle * j, up) * fw2).normalized;
                }
            }
        }

        public void UpdateViewMatrix(float maxDistance = 500.0f)
        {
            for (int i = 0; i < verticalResolution; i++)
            {
                for (int j = 0; j < horizontalResolution; j++)
                {
                    RaycastHit hitinfo;
                    if (Physics.Raycast(raysMatrix[i, j], out hitinfo, maxDistance))
                    {
                        string objname = hitinfo.collider.gameObject.name;
                        if (objname.Equals("Terrain")){
                                viewMatrix[i, j] = -1;
                        } else if (objname.StartsWith("maze1")){
                                viewMatrix[i, j] = 50;
                        } else {
                                objname = hitinfo.collider.gameObject.tag;
                                if (objname.Equals("Fire"))
                                {
                                    viewMatrix[i, j] = -50;
                                }
                                else if (objname.Equals("Life"))
                                {
                                    viewMatrix[i, j] = 250;
                                }  else if (objname.Equals("IAAgent"))
                                {
                                    viewMatrix[i,j] = 20;
                                }
                                else
                                {
                                    viewMatrix[i, j] = 0;
                                }
                        }
                    }
                    else
                    {
                        viewMatrix[i, j] = 0;
                    }
                }
            }
        }
    }
}
