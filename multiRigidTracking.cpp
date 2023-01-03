/*
Copyright © 2012 NaturalPoint Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*

SampleClient.cpp

This program connects to a NatNet server, receives a data stream, and writes that data stream
to an ascii file.  The purpose is to illustrate using the NatNetClient class.

Usage [optional]:

        SampleClient [ServerIP] [LocalIP] [OutputFilename]

        [ServerIP]			IP address of the server (e.g. 192.168.0.107) ( defaults to local machine)
        [OutputFilename]	Name of points file (pts) to write out.  defaults to Client-output.pts

*/
////////////////////////////// obj loader
#include <GL/freeglut.h>
#include <GL/glext.h>
#include <time.h>

#include <fstream>
//////////////////////////////
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <string.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <eigen3/Eigen/Geometry>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <iostream>
#include <map>
#include <memory>  // unique_ptr
#include <string>
#include <utility>

#include "omp.h"  // parallel for loop (using libs=-lgomp and CXXFLAGS=-fopenmp)
////////////////////////////////////////////////////////////////////////
#include <inttypes.h>
#include <string.h>

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

#include <NatNetCAPI.h>
#include <NatNetClient.h>
#include <NatNetTypes.h>

#include <list>
#include <thread>  // multithreading (using libs=-lpthread)
#include <vector>

#ifndef _WIN32
char getch();
#endif

class Model {
   private:
    class Face {
       public:
        int edge;
        int* vertices;
        int* texcoords;
        int normal;

        Face(int edge, int* vertices, int* texcoords, int normal = -1) {
            this->edge = edge;
            this->vertices = vertices;
            this->texcoords = texcoords;
            this->normal = normal;
        }
    };
    std::vector<float*> vertices;
    std::vector<float*> texcoords;
    std::vector<float*> normals;
    std::vector<Face> faces;
    GLuint list;

   public:
    ~Model() {
    }
    void load(const char* filename) {
        std::string line;
        std::vector<std::string> lines;
        std::ifstream in(filename);
        if (!in.is_open()) {
            printf("Cannot load model %s\n", filename);
            return;
        }
        while (!in.eof()) {
            std::getline(in, line);
            lines.push_back(line);
        }
        in.close();
        float a, b, c;
        for (std::string& line : lines) {
            if (line[0] == 'v') {
                if (line[1] == ' ') {
                    sscanf(line.c_str(), "v %f %f %f", &a, &b, &c);
                    vertices.push_back(new float[3]{a, b, c});
                } else if (line[1] == 't') {
                    sscanf(line.c_str(), "vt %f %f", &a, &b);
                    texcoords.push_back(new float[2]{a, b});
                } else {
                    sscanf(line.c_str(), "vn %f %f %f", &a, &b, &c);
                    normals.push_back(new float[3]{a, b, c});
                }
            } else if (line[0] == 'f') {
                int v0, v1, v2, t0, t1, t2, n;
                sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &v0, &t0, &n, &v1, &t1, &n, &v2, &t2, &n);
                int* v = new int[3]{v0 - 1, v1 - 1, v2 - 1};
                faces.push_back(Face(3, v, NULL, n - 1));
            }
        }
        list = glGenLists(1);
        glNewList(list, GL_COMPILE);
        for (Face& face : faces) {
            if (face.normal != -1)
                glNormal3fv(normals[face.normal]);
            else
                glDisable(GL_LIGHTING);
            glBegin(GL_POLYGON);
            for (int i = 0; i < face.edge; i++)
                glVertex3fv(vertices[face.vertices[i]]);
            glEnd();
            if (face.normal == -1)
                glEnable(GL_LIGHTING);
        }
        glEndList();
        // printf("Model: %s\n", filename);
        // printf("Vertices: %d\n", vertices.size());
        // printf("Texcoords: %d\n", texcoords.size());
        // printf("Normals: %d\n", normals.size());
        // printf("Faces: %d\n", faces.size());
        for (float* f : vertices)
            delete f;
        vertices.clear();
        for (float* f : texcoords)
            delete f;
        texcoords.clear();
        for (float* f : normals)
            delete f;
        normals.clear();
        faces.clear();
    }
    void draw() { glCallList(list); }
};

// Model* model;
std::unique_ptr<Model> model(new Model());

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WindowWidth = 500;
int WindowHeight = 500;
#define M_PIl 3.141592653589793238462643383279502884L

// int coor_accuracy = 6;
// int move = 0;
const int numberOfGrids = 20;
const float gridScale = 10.0f;  // 1.0f = 1 millimiter, 10.0f = 1 centimeters
const float cameraPosCoef = 1000.0f;
const Eigen::Vector3f coordinateTextOffset(50.f, 100.f, 50.f);

// render scene with light
bool isLightOn = true;
// changing the object representation mode true = solid, flase = wireframe
bool isSolidModeOn = false;
// changing ONLY objects line weight while in the wireframe mode
const float LINE_WEIGHT = 1.0f;

// Constants -------------------------------------------------------------------

#define WHEEL_UP 3
#define WHEEL_DOWN 4
#define CAMERA_DISTANCE_MIN 1.0
#define CAMERA_DISTANCE_MAX 100.0

// Global variables ------------------------------------------------------------

static GLint MouseX = 0;
static GLint MouseY = 0;

static double CameraLatitude = 45.0;
static double CameraLongitude = 25.0;
static double CameraDistance = 50.0;

static double EyeX = 150.0;
static double EyeY = 150.0;
static double EyeZ = 150.0;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int opti_run(int, char**);
void _WriteHeader(FILE* fp, sDataDescriptions* pBodyDefs);
void _WriteFrame(FILE* fp, sFrameOfMocapData* data);
void _WriteFooter(FILE* fp);
void NATNET_CALLCONV ServerDiscoveredCallback(const sNatNetDiscoveredServer* pDiscoveredServer, void* pUserContext);
void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData);  // receives data from the server
void NATNET_CALLCONV MessageHandler(Verbosity msgType, const char* msg);     // receives NatNet error messages
void resetClient();
int ConnectClient();

// static void Timer(int);
void glut_main(int, char**);
void anim();
void drawObj(Eigen::Vector3f pos, Eigen::Vector3f ang, Eigen::Vector3f col, const Eigen::Vector3f textOffset, int objID);
void addObjects(std::map<int, Eigen::Vector3f>*, std::map<int, Eigen::Vector3f>*, const Eigen::Vector3f*);
void draw_axis();
void drawText(char*, float, float, float);
void showCoordinates(Eigen::Vector3f, Eigen::Vector3f, const Eigen::Vector3f, int);
void draw_grid();
void update_camera_location();
void init_scene();
void display();
void resize(int, int);
void mouse(int, int, int, int);
void motion(int, int);

static const ConnectionType kDefaultConnectionType = ConnectionType_Multicast;

NatNetClient* g_pClient = NULL;
FILE* g_outputFile;

std::vector<sNatNetDiscoveredServer> g_discoveredServers;
sNatNetClientConnectParams g_connectParams;
char g_discoveredMulticastGroupAddr[kNatNetIpv4AddrStrLenMax] = NATNET_DEFAULT_MULTICAST_ADDRESS;
int g_analogSamplesPerMocapFrame = 0;
sServerDescription g_serverDescription;

int numberOfRigids;
std::map<int, Eigen::Vector3f> rigids_map_pos;
std::map<int, Eigen::Vector3f> rigids_map_ang;
Eigen::Vector3f colorSet[] = {Eigen::Vector3f(0.f, .4f, 1.f), Eigen::Vector3f(.4f, 1.f, .2f), Eigen::Vector3f(1.f, .2f, 0.f), Eigen::Vector3f(1.f, 0.f, 1.f), Eigen::Vector3f(1.f, 1.f, 0.f), Eigen::Vector3f(1.f, .6f, 0.f), Eigen::Vector3f(.4f, .6f, 0.f), Eigen::Vector3f(.2f, .2f, .8f), Eigen::Vector3f(.4f, 1.f, 1.f), Eigen::Vector3f(1.f, 0.f, .4f)};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    // glutInit(&argc, argv);
    // model = new Model();

    std::thread t1(glut_main, argc, argv);

    // std::thread t1(glut_main);
    std::thread t2(opti_run, argc, argv);

    // glut_main(argc, argv);

    // std::thread t3(func);

    // opti_run
    (argc, argv);
    t1.join();
    t2.join();
    // t3.join();

    // delete model;
    // model = nullptr;
    return 0;
}

void glut_main(int argc, char** argv) {
    // void glut_main() {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WindowWidth, WindowHeight);
    // glutInitWindowPosition(100, 100);
    glutCreateWindow("OptiTrack App");

    glutDisplayFunc(display);
    glutReshapeFunc(resize);

    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    init_scene();
    // Timer(0);
    glutIdleFunc(anim);

    glutMainLoop();
}

int opti_run(int argc, char** argv) {
    // print version info
    unsigned char ver[4];
    NatNet_GetVersion(ver);
    printf("NatNet Sample Client (NatNet ver. %d.%d.%d.%d)\n", ver[0], ver[1], ver[2], ver[3]);

    // Install logging callback
    NatNet_SetLogCallback(MessageHandler);

    // create NatNet client
    g_pClient = new NatNetClient();

    printf("CREATE CLIENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

    // set the frame callback handler
    g_pClient->SetFrameReceivedCallback(DataHandler, g_pClient);  // this function will receive data from the server

    // If no arguments were specified on the command line,
    // attempt to discover servers on the local network.
    if (argc == 1) {
        // An example of synchronous server discovery.
#if 0
        const unsigned int kDiscoveryWaitTimeMillisec = 5 * 1000; // Wait 5 seconds for responses.
        const int kMaxDescriptions = 10; // Get info for, at most, the first 10 servers to respond.
        sNatNetDiscoveredServer servers[kMaxDescriptions];
        int actualNumDescriptions = kMaxDescriptions;
        NatNet_BroadcastServerDiscovery( servers, &actualNumDescriptions );

        if ( actualNumDescriptions < kMaxDescriptions )
        {
            // If this happens, more servers responded than the array was able to store.
        }
#endif

        // Do asynchronous server discovery.
        printf("Looking for servers on the local network.\n");
        printf("Press the number key that corresponds to any discovered server to connect to that server.\n");
        printf("Press Q at any time to quit.\n\n");

        NatNetDiscoveryHandle discovery;
        NatNet_CreateAsyncServerDiscovery(&discovery, ServerDiscoveredCallback);

        while (const int c = getch()) {
            if (c >= '1' && c <= '9') {
                const size_t serverIndex = c - '1';
                if (serverIndex < g_discoveredServers.size()) {
                    const sNatNetDiscoveredServer& discoveredServer = g_discoveredServers[serverIndex];

                    if (discoveredServer.serverDescription.bConnectionInfoValid) {
                        // Build the connection parameters.
#ifdef _WIN32
                        _snprintf_s(
#else
                        snprintf(
#endif
                            g_discoveredMulticastGroupAddr, sizeof g_discoveredMulticastGroupAddr,
                            "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 "",
                            discoveredServer.serverDescription.ConnectionMulticastAddress[0],
                            discoveredServer.serverDescription.ConnectionMulticastAddress[1],
                            discoveredServer.serverDescription.ConnectionMulticastAddress[2],
                            discoveredServer.serverDescription.ConnectionMulticastAddress[3]);

                        g_connectParams.connectionType = discoveredServer.serverDescription.ConnectionMulticast ? ConnectionType_Multicast : ConnectionType_Unicast;
                        g_connectParams.serverCommandPort = discoveredServer.serverCommandPort;
                        g_connectParams.serverDataPort = discoveredServer.serverDescription.ConnectionDataPort;
                        g_connectParams.serverAddress = discoveredServer.serverAddress;
                        g_connectParams.localAddress = discoveredServer.localAddress;
                        g_connectParams.multicastAddress = g_discoveredMulticastGroupAddr;
                    } else {
                        // We're missing some info because it's a legacy server.
                        // Guess the defaults and make a best effort attempt to connect.
                        g_connectParams.connectionType = kDefaultConnectionType;
                        g_connectParams.serverCommandPort = discoveredServer.serverCommandPort;
                        g_connectParams.serverDataPort = 0;
                        g_connectParams.serverAddress = discoveredServer.serverAddress;
                        g_connectParams.localAddress = discoveredServer.localAddress;
                        g_connectParams.multicastAddress = NULL;
                    }

                    break;
                }
            } else if (c == 'q') {
                return 0;
            }
        }

        NatNet_FreeAsyncServerDiscovery(discovery);
    } else {
        g_connectParams.connectionType = kDefaultConnectionType;

        if (argc >= 2) {
            g_connectParams.serverAddress = argv[1];
        }

        if (argc >= 3) {
            g_connectParams.localAddress = argv[2];
        }
    }

    int iResult;

    // Connect to Motive
    iResult = ConnectClient();
    if (iResult != ErrorCode_OK) {
        printf("Error initializing client. See log for details. Exiting.\n");
        return 1;
    } else {
        printf("Client initialized and ready.\n");
    }

    // Send/receive test request
    void* response;
    int nBytes;
    printf("[SampleClient] Sending Test Request\n");
    iResult = g_pClient->SendMessageAndWait("TestRequest", &response, &nBytes);
    if (iResult == ErrorCode_OK) {
        printf("[SampleClient] Received: %s\n", (char*)response);
    }

    // Retrieve Data Descriptions from Motive
    printf("\n\n[SampleClient] Requesting Data Descriptions...\n");
    sDataDescriptions* pDataDefs = NULL;
    iResult = g_pClient->GetDataDescriptionList(&pDataDefs);
    if (iResult != ErrorCode_OK || pDataDefs == NULL) {
        printf("[SampleClient] Unable to retrieve Data Descriptions.\n");
    } else {
        printf("[SampleClient] Received %d Data Descriptions:\n", pDataDefs->nDataDescriptions);
        for (int i = 0; i < pDataDefs->nDataDescriptions; i++) {
            printf("Data Description # %d (type=%d)\n", i, pDataDefs->arrDataDescriptions[i].type);
            if (pDataDefs->arrDataDescriptions[i].type == Descriptor_MarkerSet) {
                // MarkerSet
                sMarkerSetDescription* pMS = pDataDefs->arrDataDescriptions[i].Data.MarkerSetDescription;
                printf("MarkerSet Name : %s\n", pMS->szName);
                for (int i = 0; i < pMS->nMarkers; i++)
                    printf("%s\n", pMS->szMarkerNames[i]);

            } else if (pDataDefs->arrDataDescriptions[i].type == Descriptor_RigidBody) {
                // RigidBody
                sRigidBodyDescription* pRB = pDataDefs->arrDataDescriptions[i].Data.RigidBodyDescription;
                printf("RigidBody Name : %s\n", pRB->szName);
                printf("RigidBody ID : %d\n", pRB->ID);
                printf("RigidBody Parent ID : %d\n", pRB->parentID);
                printf("Parent Offset : %3.2f,%3.2f,%3.2f\n", pRB->offsetx, pRB->offsety, pRB->offsetz);

                if (pRB->MarkerPositions != NULL && pRB->MarkerRequiredLabels != NULL) {
                    for (int markerIdx = 0; markerIdx < pRB->nMarkers; ++markerIdx) {
                        const MarkerData& markerPosition = pRB->MarkerPositions[markerIdx];
                        const int markerRequiredLabel = pRB->MarkerRequiredLabels[markerIdx];

                        printf("\tMarker #%d:\n", markerIdx);
                        printf("\t\tPosition: %.2f, %.2f, %.2f\n", markerPosition[0], markerPosition[1], markerPosition[2]);

                        if (markerRequiredLabel != 0) {
                            printf("\t\tRequired active label: %d\n", markerRequiredLabel);
                        }
                    }
                }
            } else if (pDataDefs->arrDataDescriptions[i].type == Descriptor_Skeleton) {
                // Skeleton
                sSkeletonDescription* pSK = pDataDefs->arrDataDescriptions[i].Data.SkeletonDescription;
                printf("Skeleton Name : %s\n", pSK->szName);
                printf("Skeleton ID : %d\n", pSK->skeletonID);
                printf("RigidBody (Bone) Count : %d\n", pSK->nRigidBodies);
                for (int j = 0; j < pSK->nRigidBodies; j++) {
                    sRigidBodyDescription* pRB = &pSK->RigidBodies[j];
                    printf("  RigidBody Name : %s\n", pRB->szName);
                    printf("  RigidBody ID : %d\n", pRB->ID);
                    printf("  RigidBody Parent ID : %d\n", pRB->parentID);
                    printf("  Parent Offset : %3.2f,%3.2f,%3.2f\n", pRB->offsetx, pRB->offsety, pRB->offsetz);
                }
            } else if (pDataDefs->arrDataDescriptions[i].type == Descriptor_ForcePlate) {
                // Force Plate
                sForcePlateDescription* pFP = pDataDefs->arrDataDescriptions[i].Data.ForcePlateDescription;
                printf("Force Plate ID : %d\n", pFP->ID);
                printf("Force Plate Serial : %s\n", pFP->strSerialNo);
                printf("Force Plate Width : %3.2f\n", pFP->fWidth);
                printf("Force Plate Length : %3.2f\n", pFP->fLength);
                printf("Force Plate Electrical Center Offset (%3.3f, %3.3f, %3.3f)\n", pFP->fOriginX, pFP->fOriginY, pFP->fOriginZ);
                for (int iCorner = 0; iCorner < 4; iCorner++)
                    printf("Force Plate Corner %d : (%3.4f, %3.4f, %3.4f)\n", iCorner, pFP->fCorners[iCorner][0], pFP->fCorners[iCorner][1], pFP->fCorners[iCorner][2]);
                printf("Force Plate Type : %d\n", pFP->iPlateType);
                printf("Force Plate Data Type : %d\n", pFP->iChannelDataType);
                printf("Force Plate Channel Count : %d\n", pFP->nChannels);
                for (int iChannel = 0; iChannel < pFP->nChannels; iChannel++)
                    printf("\tChannel %d : %s\n", iChannel, pFP->szChannelNames[iChannel]);
            } else if (pDataDefs->arrDataDescriptions[i].type == Descriptor_Device) {
                // Peripheral Device
                sDeviceDescription* pDevice = pDataDefs->arrDataDescriptions[i].Data.DeviceDescription;
                printf("Device Name : %s\n", pDevice->strName);
                printf("Device Serial : %s\n", pDevice->strSerialNo);
                printf("Device ID : %d\n", pDevice->ID);
                printf("Device Channel Count : %d\n", pDevice->nChannels);
                for (int iChannel = 0; iChannel < pDevice->nChannels; iChannel++)
                    printf("\tChannel %d : %s\n", iChannel, pDevice->szChannelNames[iChannel]);
            } else if (pDataDefs->arrDataDescriptions[i].type == Descriptor_Camera) {
                // Camera
                sCameraDescription* pCamera = pDataDefs->arrDataDescriptions[i].Data.CameraDescription;
                printf("Camera Name : %s\n", pCamera->strName);
                printf("Camera Position (%3.2f, %3.2f, %3.2f)\n", pCamera->x, pCamera->y, pCamera->z);
                printf("Camera Orientation (%3.2f, %3.2f, %3.2f, %3.2f)\n", pCamera->qx, pCamera->qy, pCamera->qz, pCamera->qw);
            } else {
                printf("Unknown data type.\n");
                // Unknown
            }
        }
    }

    // Create data file for writing received stream into
    const char* szFile = "Client-output.pts";
    if (argc > 3)
        szFile = argv[3];

    g_outputFile = fopen(szFile, "w");
    if (!g_outputFile) {
        printf("Error opening output file %s.  Exiting.\n", szFile);
        exit(1);
    }

    if (pDataDefs) {
        _WriteHeader(g_outputFile, pDataDefs);
        NatNet_FreeDescriptions(pDataDefs);
        pDataDefs = NULL;
    }

    // Ready to receive marker stream!
    printf("\nClient is connected to server and listening for data...\n");
    bool bExit = false;
    while (const int c = getch()) {
        switch (c) {
            case 'q':
                bExit = true;
                break;
            case 'r':
                resetClient();
                break;
            case 'p':
                sServerDescription ServerDescription;
                memset(&ServerDescription, 0, sizeof(ServerDescription));
                g_pClient->GetServerDescription(&ServerDescription);
                if (!ServerDescription.HostPresent) {
                    printf("Unable to connect to server. Host not present. Exiting.");
                    return 1;
                }
                break;
            case 's': {
                printf("\n\n[SampleClient] Requesting Data Descriptions...");
                sDataDescriptions* pDataDefs = NULL;
                iResult = g_pClient->GetDataDescriptionList(&pDataDefs);
                if (iResult != ErrorCode_OK || pDataDefs == NULL) {
                    printf("[SampleClient] Unable to retrieve Data Descriptions.");
                } else {
                    printf("[SampleClient] Received %d Data Descriptions:\n", pDataDefs->nDataDescriptions);
                }
            } break;
            case 'm':  // change to multicast
                g_connectParams.connectionType = ConnectionType_Multicast;
                iResult = ConnectClient();
                if (iResult == ErrorCode_OK)
                    printf("Client connection type changed to Multicast.\n\n");
                else
                    printf("Error changing client connection type to Multicast.\n\n");
                break;
            case 'u':  // change to unicast
                g_connectParams.connectionType = ConnectionType_Unicast;
                iResult = ConnectClient();
                if (iResult == ErrorCode_OK)
                    printf("Client connection type changed to Unicast.\n\n");
                else
                    printf("Error changing client connection type to Unicast.\n\n");
                break;
            case 'c':  // connect
                iResult = ConnectClient();
                break;
            case 'd':  // disconnect
                // note: applies to unicast connections only - indicates to Motive to stop sending packets to that client endpoint
                iResult = g_pClient->SendMessageAndWait("Disconnect", &response, &nBytes);
                if (iResult == ErrorCode_OK)
                    printf("[SampleClient] Disconnected");
                break;
            default:
                break;
        }
        if (bExit)
            break;
    }

    // Done - clean up.
    if (g_pClient) {
        g_pClient->Disconnect();
        delete g_pClient;
        g_pClient = NULL;
    }

    if (g_outputFile) {
        _WriteFooter(g_outputFile);
        fclose(g_outputFile);
        g_outputFile = NULL;
    }

    return ErrorCode_OK;
}

void NATNET_CALLCONV ServerDiscoveredCallback(const sNatNetDiscoveredServer* pDiscoveredServer, void* pUserContext) {
    char serverHotkey = '.';
    if (g_discoveredServers.size() < 9) {
        serverHotkey = static_cast<char>('1' + g_discoveredServers.size());
    }

    printf("[%c] %s %d.%d at %s ",
           serverHotkey,
           pDiscoveredServer->serverDescription.szHostApp,
           pDiscoveredServer->serverDescription.HostAppVersion[0],
           pDiscoveredServer->serverDescription.HostAppVersion[1],
           pDiscoveredServer->serverAddress);

    if (pDiscoveredServer->serverDescription.bConnectionInfoValid) {
        printf("(%s)\n", pDiscoveredServer->serverDescription.ConnectionMulticast ? "multicast" : "unicast");
    } else {
        printf("(WARNING: Legacy server, could not autodetect settings. Auto-connect may not work reliably.)\n");
    }

    g_discoveredServers.push_back(*pDiscoveredServer);
}

// Establish a NatNet Client connection
int ConnectClient() {
    // Release previous server
    g_pClient->Disconnect();

    // Init Client and connect to NatNet server
    int retCode = g_pClient->Connect(g_connectParams);
    if (retCode != ErrorCode_OK) {
        printf("Unable to connect to server.  Error code: %d. Exiting.\n", retCode);
        return ErrorCode_Internal;
    } else {
        // connection succeeded

        void* pResult;
        int nBytes = 0;
        ErrorCode ret = ErrorCode_OK;

        // print server info
        memset(&g_serverDescription, 0, sizeof(g_serverDescription));
        ret = g_pClient->GetServerDescription(&g_serverDescription);
        if (ret != ErrorCode_OK || !g_serverDescription.HostPresent) {
            printf("Unable to connect to server. Host not present. Exiting.\n");
            return 1;
        }
        printf("\n[SampleClient] Server application info:\n");
        printf("Application: %s (ver. %d.%d.%d.%d)\n", g_serverDescription.szHostApp, g_serverDescription.HostAppVersion[0],
               g_serverDescription.HostAppVersion[1], g_serverDescription.HostAppVersion[2], g_serverDescription.HostAppVersion[3]);
        printf("NatNet Version: %d.%d.%d.%d\n", g_serverDescription.NatNetVersion[0], g_serverDescription.NatNetVersion[1],
               g_serverDescription.NatNetVersion[2], g_serverDescription.NatNetVersion[3]);
        printf("Client IP:%s\n", g_connectParams.localAddress);
        printf("Server IP:%s\n", g_connectParams.serverAddress);
        printf("Server Name:%s\n", g_serverDescription.szHostComputerName);

        // get mocap frame rate
        ret = g_pClient->SendMessageAndWait("FrameRate", &pResult, &nBytes);

        if (ret == ErrorCode_OK) {
            float fRate = *((float*)pResult);
            printf("Mocap Framerate : %3.2f\n", fRate);
        } else
            printf("Error getting frame rate.\n");

        // get # of analog samples per mocap frame of data
        ret = g_pClient->SendMessageAndWait("AnalogSamplesPerMocapFrame", &pResult, &nBytes);
        if (ret == ErrorCode_OK) {
            g_analogSamplesPerMocapFrame = *((int*)pResult);
            printf("Analog Samples Per Mocap Frame : %d\n", g_analogSamplesPerMocapFrame);
        } else
            printf("Error getting Analog frame rate.\n");
    }

    return ErrorCode_OK;
}

// DataHandler receives data from the server
// This function is called by NatNet when a frame of mocap data is available
void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData) {
    NatNetClient* pClient = (NatNetClient*)pUserData;

    // Software latency here is defined as the span of time between:
    //   a) The reception of a complete group of 2D frames from the camera system (CameraDataReceivedTimestamp)
    // and
    //   b) The time immediately prior to the NatNet frame being transmitted over the network (TransmitTimestamp)
    //
    // This figure may appear slightly higher than the "software latency" reported in the Motive user interface,
    // because it additionally includes the time spent preparing to stream the data via NatNet.
    const uint64_t softwareLatencyHostTicks = data->TransmitTimestamp - data->CameraDataReceivedTimestamp;
    const double softwareLatencyMillisec = (softwareLatencyHostTicks * 1000) / static_cast<double>(g_serverDescription.HighResClockFrequency);

    // Transit latency is defined as the span of time between Motive transmitting the frame of data, and its reception by the client (now).
    // The SecondsSinceHostTimestamp method relies on NatNetClient's internal clock synchronization with the server using Cristian's algorithm.
    const double transitLatencyMillisec = pClient->SecondsSinceHostTimestamp(data->TransmitTimestamp) * 1000.0;

    if (g_outputFile) {
        _WriteFrame(g_outputFile, data);
    }

    int i = 0;

    printf("FrameID : %d\n", data->iFrame);
    printf("Timestamp : %3.2lf\n", data->fTimestamp);
    printf("Software latency : %.2lf milliseconds\n", softwareLatencyMillisec);

    // Only recent versions of the Motive software in combination with ethernet camera systems support system latency measurement.
    // If it's unavailable (for example, with USB camera systems, or during playback), this field will be zero.
    const bool bSystemLatencyAvailable = data->CameraMidExposureTimestamp != 0;

    if (bSystemLatencyAvailable) {
        // System latency here is defined as the span of time between:
        //   a) The midpoint of the camera exposure window, and therefore the average age of the photons (CameraMidExposureTimestamp)
        // and
        //   b) The time immediately prior to the NatNet frame being transmitted over the network (TransmitTimestamp)
        const uint64_t systemLatencyHostTicks = data->TransmitTimestamp - data->CameraMidExposureTimestamp;
        const double systemLatencyMillisec = (systemLatencyHostTicks * 1000) / static_cast<double>(g_serverDescription.HighResClockFrequency);

        // Client latency is defined as the sum of system latency and the transit time taken to relay the data to the NatNet client.
        // This is the all-inclusive measurement (photons to client processing).
        const double clientLatencyMillisec = pClient->SecondsSinceHostTimestamp(data->CameraMidExposureTimestamp) * 1000.0;

        // You could equivalently do the following (not accounting for time elapsed since we calculated transit latency above):
        // const double clientLatencyMillisec = systemLatencyMillisec + transitLatencyMillisec;

        printf("System latency : %.2lf milliseconds\n", systemLatencyMillisec);
        printf("Total client latency : %.2lf milliseconds (transit time +%.2lf ms)\n", clientLatencyMillisec, transitLatencyMillisec);
    } else {
        printf("Transit latency : %.2lf milliseconds\n", transitLatencyMillisec);
    }

    // FrameOfMocapData params
    bool bIsRecording = ((data->params & 0x01) != 0);
    bool bTrackedModelsChanged = ((data->params & 0x02) != 0);
    if (bIsRecording)
        printf("RECORDING\n");
    if (bTrackedModelsChanged)
        printf("Models Changed.\n");

    // timecode - for systems with an eSync and SMPTE timecode generator - decode to values
    int hour, minute, second, frame, subframe;
    NatNet_DecodeTimecode(data->Timecode, data->TimecodeSubframe, &hour, &minute, &second, &frame, &subframe);
    // decode to friendly string
    char szTimecode[128] = "";
    NatNet_TimecodeStringify(data->Timecode, data->TimecodeSubframe, szTimecode, 128);
    printf("Timecode : %s\n", szTimecode);

    // Rigid Bodies
    numberOfRigids = data->nRigidBodies;
    printf("Rigid Bodies [Count=%d]\n", data->nRigidBodies);
    for (i = 0; i < data->nRigidBodies; i++) {
        // params
        // 0x01 : bool, rigid body was successfully tracked in this frame
        bool bTrackingValid = data->RigidBodies[i].params & 0x01;

        printf("Rigid Body [ID=%d  Error=%3.2f  Valid=%d]\n", data->RigidBodies[i].ID, data->RigidBodies[i].MeanError, bTrackingValid);
        printf("\tx\ty\tz\tqx\tqy\tqz\tqw\n");
        printf("\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\n",
               data->RigidBodies[i].x,
               data->RigidBodies[i].y,
               data->RigidBodies[i].z,
               data->RigidBodies[i].qx,
               data->RigidBodies[i].qy,
               data->RigidBodies[i].qz,
               data->RigidBodies[i].qw);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        Eigen::Vector3f current_pos(data->RigidBodies[i].x, data->RigidBodies[i].y, data->RigidBodies[i].z);
        rigids_map_pos[data->RigidBodies[i].ID] = current_pos;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // converting the Quaternion rotation to the Euler rotation
        Eigen::Quaternionf qq;
        qq.x() = data->RigidBodies[i].qx;
        qq.y() = data->RigidBodies[i].qy;
        qq.z() = data->RigidBodies[i].qz;
        qq.w() = data->RigidBodies[i].qw;

        Eigen::Vector3f euler = qq.toRotationMatrix().eulerAngles(0, 1, 2);
        euler = euler * 180 / M_PI;
        // std::cout << "Euler from quaternion in roll, pitch, yaw" << std::endl << euler << std::endl;
        rigids_map_ang[data->RigidBodies[i].ID] = euler;
    }

    // Skeletons
    printf("Skeletons [Count=%d]\n", data->nSkeletons);
    for (i = 0; i < data->nSkeletons; i++) {
        sSkeletonData skData = data->Skeletons[i];
        printf("Skeleton [ID=%d  Bone count=%d]\n", skData.skeletonID, skData.nRigidBodies);
        for (int j = 0; j < skData.nRigidBodies; j++) {
            sRigidBodyData rbData = skData.RigidBodyData[j];
            printf("Bone %d\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\n",
                   rbData.ID, rbData.x, rbData.y, rbData.z, rbData.qx, rbData.qy, rbData.qz, rbData.qw);
        }
    }

    // labeled markers - this includes all markers (Active, Passive, and 'unlabeled' (markers with no asset but a PointCloud ID)
    bool bOccluded;      // marker was not visible (occluded) in this frame
    bool bPCSolved;      // reported position provided by point cloud solve
    bool bModelSolved;   // reported position provided by model solve
    bool bHasModel;      // marker has an associated asset in the data stream
    bool bUnlabeled;     // marker is 'unlabeled', but has a point cloud ID that matches Motive PointCloud ID (In Motive 3D View)
    bool bActiveMarker;  // marker is an actively labeled LED marker

    printf("Markers [Count=%d]\n", data->nLabeledMarkers);
    for (i = 0; i < data->nLabeledMarkers; i++) {
        bOccluded = ((data->LabeledMarkers[i].params & 0x01) != 0);
        bPCSolved = ((data->LabeledMarkers[i].params & 0x02) != 0);
        bModelSolved = ((data->LabeledMarkers[i].params & 0x04) != 0);
        bHasModel = ((data->LabeledMarkers[i].params & 0x08) != 0);
        bUnlabeled = ((data->LabeledMarkers[i].params & 0x10) != 0);
        bActiveMarker = ((data->LabeledMarkers[i].params & 0x20) != 0);

        sMarker marker = data->LabeledMarkers[i];

        // Marker ID Scheme:
        // Active Markers:
        //   ID = ActiveID, correlates to RB ActiveLabels list
        // Passive Markers:
        //   If Asset with Legacy Labels
        //      AssetID 	(Hi Word)
        //      MemberID	(Lo Word)
        //   Else
        //      PointCloud ID
        int modelID, markerID;
        NatNet_DecodeID(marker.ID, &modelID, &markerID);

        char szMarkerType[512];
        if (bActiveMarker)
            strcpy(szMarkerType, "Active");
        else if (bUnlabeled)
            strcpy(szMarkerType, "Unlabeled");
        else
            strcpy(szMarkerType, "Labeled");

        printf("%s Marker [ModelID=%d, MarkerID=%d] [size=%3.2f] [pos=%3.2f,%3.2f,%3.2f]\n",
               szMarkerType, modelID, markerID, marker.size, marker.x, marker.y, marker.z);
    }

    // force plates
    printf("Force Plate [Count=%d]\n", data->nForcePlates);
    for (int iPlate = 0; iPlate < data->nForcePlates; iPlate++) {
        printf("Force Plate %d\n", data->ForcePlates[iPlate].ID);
        for (int iChannel = 0; iChannel < data->ForcePlates[iPlate].nChannels; iChannel++) {
            printf("\tChannel %d:\t", iChannel);
            if (data->ForcePlates[iPlate].ChannelData[iChannel].nFrames == 0) {
                printf("\tEmpty Frame\n");
            } else if (data->ForcePlates[iPlate].ChannelData[iChannel].nFrames != g_analogSamplesPerMocapFrame) {
                printf("\tPartial Frame [Expected:%d   Actual:%d]\n", g_analogSamplesPerMocapFrame, data->ForcePlates[iPlate].ChannelData[iChannel].nFrames);
            }
            for (int iSample = 0; iSample < data->ForcePlates[iPlate].ChannelData[iChannel].nFrames; iSample++)
                printf("%3.2f\t", data->ForcePlates[iPlate].ChannelData[iChannel].Values[iSample]);
            printf("\n");
        }
    }

    // devices
    printf("Device [Count=%d]\n", data->nDevices);
    for (int iDevice = 0; iDevice < data->nDevices; iDevice++) {
        printf("Device %d\n", data->Devices[iDevice].ID);
        for (int iChannel = 0; iChannel < data->Devices[iDevice].nChannels; iChannel++) {
            printf("\tChannel %d:\t", iChannel);
            if (data->Devices[iDevice].ChannelData[iChannel].nFrames == 0) {
                printf("\tEmpty Frame\n");
            } else if (data->Devices[iDevice].ChannelData[iChannel].nFrames != g_analogSamplesPerMocapFrame) {
                printf("\tPartial Frame [Expected:%d   Actual:%d]\n", g_analogSamplesPerMocapFrame, data->Devices[iDevice].ChannelData[iChannel].nFrames);
            }
            for (int iSample = 0; iSample < data->Devices[iDevice].ChannelData[iChannel].nFrames; iSample++)
                printf("%3.2f\t", data->Devices[iDevice].ChannelData[iChannel].Values[iSample]);
            printf("\n");
        }
    }
}

// MessageHandler receives NatNet error/debug messages
void NATNET_CALLCONV MessageHandler(Verbosity msgType, const char* msg) {
    // Optional: Filter out debug messages
    if (msgType < Verbosity_Info) {
        return;
    }

    printf("\n[NatNetLib]");

    switch (msgType) {
        case Verbosity_Debug:
            printf(" [DEBUG]");
            break;
        case Verbosity_Info:
            printf("  [INFO]");
            break;
        case Verbosity_Warning:
            printf("  [WARN]");
            break;
        case Verbosity_Error:
            printf(" [ERROR]");
            break;
        default:
            printf(" [?????]");
            break;
    }

    printf(": %s\n", msg);
}

/* File writing routines */
void _WriteHeader(FILE* fp, sDataDescriptions* pBodyDefs) {
    int i = 0;

    if (pBodyDefs->arrDataDescriptions[0].type != Descriptor_MarkerSet)
        return;

    sMarkerSetDescription* pMS = pBodyDefs->arrDataDescriptions[0].Data.MarkerSetDescription;

    fprintf(fp, "<MarkerSet>\n\n");
    fprintf(fp, "<Name>\n%s\n</Name>\n\n", pMS->szName);

    fprintf(fp, "<Markers>\n");
    for (i = 0; i < pMS->nMarkers; i++) {
        fprintf(fp, "%s\n", pMS->szMarkerNames[i]);
    }
    fprintf(fp, "</Markers>\n\n");

    fprintf(fp, "<Data>\n");
    fprintf(fp, "Frame#\t");
    for (i = 0; i < pMS->nMarkers; i++) {
        fprintf(fp, "M%dX\tM%dY\tM%dZ\t", i, i, i);
    }
    fprintf(fp, "\n");
}

void _WriteFrame(FILE* fp, sFrameOfMocapData* data) {
    fprintf(fp, "%d", data->iFrame);
    for (int i = 0; i < data->MocapData->nMarkers; i++) {
        fprintf(fp, "\t%.5f\t%.5f\t%.5f", data->MocapData->Markers[i][0], data->MocapData->Markers[i][1], data->MocapData->Markers[i][2]);
    }
    fprintf(fp, "\n");
}

void _WriteFooter(FILE* fp) {
    fprintf(fp, "</Data>\n\n");
    fprintf(fp, "</MarkerSet>\n");
}

void resetClient() {
    int iSuccess;

    printf("\n\nre-setting Client\n\n.");

    iSuccess = g_pClient->Disconnect();
    if (iSuccess != 0)
        printf("error un-initting Client\n");

    iSuccess = g_pClient->Connect(g_connectParams);
    if (iSuccess != 0)
        printf("error re-initting Client\n");
}

#ifndef _WIN32
char getch() {
    char buf = 0;
    termios old = {0};

    fflush(stdout);

    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");

    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;

    if (tcsetattr(0, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");

    if (read(0, &buf, 1) < 0)
        perror("read()");

    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;

    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror("tcsetattr ~ICANON");

    // printf( "%c\n", buf );

    return buf;
}
#endif
//////////////////////////////////////////////////////////////////////////// start working on integrating glut with the code to see if it shows the spheres at the poit locations
// static void Timer(int value) {
//     move += 0.1;
//     glutPostRedisplay();
//     // 100 milliseconds
//     glutTimerFunc(100, Timer, 0);
// }

void anim() {
    glutPostRedisplay();
}

// void init() {
//     glClearColor(1.0, 1.0, 1.0, 1.0);
// }

void draw_grid() {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glPushAttrib(GL_LIGHTING_BIT);
    {
        glDisable(GL_LIGHTING);
        // glColor3f(0.4, 0.4, 0.4);
        glScalef(gridScale * 2, gridScale * 2, gridScale * 2);
        for (int k = 0; k < 2; k++) {
            for (int i = -numberOfGrids; i < numberOfGrids + 1; i++) {
                if (!(i % 10))
                    glColor3f(.8, .8, .8);
                else
                    glColor3f(0.4, 0.4, 0.4);
                glBegin(GL_LINE_STRIP);
                {
                    for (int j = -numberOfGrids; j < numberOfGrids + 1; j++)
                        glVertex3f(k == 0 ? i : j, 0, k == 0 ? j : i);
                }
                glEnd();
            }
        }
    }
    glPopAttrib();
    glPopMatrix();
}

// Update camera ---------------------------------------------------------------

void update_camera_location() {
    // Based on some magical math:
    // http://en.wikipedia.org/wiki/List_of_canonical_coordinate_transformations#From_spherical_coordinates
    // and some help from Dr. John Stewman

    double L = CameraDistance * std::cos(M_PI * CameraLongitude / 180.0);

    EyeX = L * -std::sin(M_PI * CameraLatitude / 180.0);
    EyeY = CameraDistance * std::sin(M_PI * CameraLongitude / 180.0);
    EyeZ = L * std::cos(M_PI * CameraLatitude / 180.0);

    glutPostRedisplay();
}

// Initialize scene ------------------------------------------------------------

void init_scene() {
    // glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (GLdouble)WindowWidth / (GLdouble)WindowHeight, 1.0, 750.0);
    glMatrixMode(GL_MODELVIEW);

    if (isLightOn) {
        // adding light
        GLfloat diffuse0[] = {0.8, 0.8, 0.8, 1.0};
        GLfloat ambient0[] = {0.2, 0.2, 0.2, 1.0};
        GLfloat specular0[] = {1.0, 1.0, 1.0, 1.0};
        GLfloat position0[] = {100.0, 100.0, 100.0, 1.0};

        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);
        glLightfv(GL_LIGHT0, GL_POSITION, position0);

        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        glEnable(GL_NORMALIZE);
        glEnable(GL_SMOOTH);
        // glPolygonMode(GL_FRONT, GL_FILL);
        // glPolygonMode(GL_BACK, GL_FILL);

        glShadeModel(GL_SMOOTH);
        // glShadeModel(GL_FLAT);
        // glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);

        if (isSolidModeOn) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glEnable(GL_CULL_FACE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
    }

    model->load("Models/mmm.obj");
    // update camera location
    update_camera_location();
}

void drawObj(Eigen::Vector3f pos, Eigen::Vector3f ang, Eigen::Vector3f color, const Eigen::Vector3f textOffset, int objID) {
    glColor3f(color[0], color[1], color[2]);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    // translation matrix (the reason that we use minus x and z is the in OpenGL the Z axes is towards the screen and the X axes is to the right, despite the Optitrack)
    glTranslatef(-pos[0] * cameraPosCoef, pos[1] * cameraPosCoef, -pos[2] * cameraPosCoef);
    // rotation matrices
    glRotatef(180 - ang[0], 1, 0, 0);
    glRotatef(180 - ang[1], 0, 1, 0);
    glRotatef(180 - ang[2], 0, 0, 1);
    glScalef(100, 100, 100);
    if (!isSolidModeOn) glLineWidth(LINE_WEIGHT);
    // glutSolidCube(100);
    model->draw();
    glLineWidth(.5);
    showCoordinates(pos, ang, textOffset, objID);
    glPopMatrix();
}

void addObjects(std::map<int, Eigen::Vector3f>* rigids_map_pos, std::map<int, Eigen::Vector3f>* rigids_map_ang, const Eigen::Vector3f* coorTextOffset) {
    // int th_id;
#pragma omp parallel
    for (int i = 1; i <= numberOfRigids; i++) {
        // Eigen::Vector3f color(static_cast<float>(i) / static_cast<float>(numberOfRigids), 1.0 - static_cast<float>(i) / static_cast<float>(numberOfRigids), static_cast<float>(i) / 10.0);
        Eigen::Vector3f color = colorSet[i - 1];
        // th_id = omp_get_thread_num();
        // printf("=> thread id: %i\t", th_id);
        drawObj((*rigids_map_pos)[i], (*rigids_map_ang)[i], color, *coorTextOffset, i);
    }
}

void draw_axis() {
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glLineWidth(4.0);
    glBegin(GL_LINES);
    {
        glColor3f(1.0, 0.0, 0.0);  // X Axis
        glVertex3f(-WindowWidth, 0.0, 0.0);
        glVertex3f(WindowWidth, 0.0, 0.0);

        glColor3f(0.0, 1.0, 0.0);  // Y Axis
        glVertex3f(0.0, -WindowHeight, 0.0);
        glVertex3f(0.0, WindowHeight, 0.0);

        glColor3f(0.0, 0.0, 1.0);  // Z Axis
        glVertex3f(0.0, 0.0, -WindowHeight);
        glVertex3f(0.0, 0.0, WindowHeight);
    }
    glEnd();
    glPopAttrib();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0);

    glLoadIdentity();
    gluLookAt(EyeX, EyeY, EyeZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    draw_axis();
    draw_grid();

    addObjects(&rigids_map_pos, &rigids_map_ang, &coordinateTextOffset);

    glutSwapBuffers();
}

void resize(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-WindowWidth, WindowWidth, -WindowHeight, WindowHeight, -WindowHeight, WindowHeight);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void drawText(char* s, float x, float y, float z) {
    // glRasterPos2f(x, y);
    glRasterPos3f(x, y, z);
    // glColor3f(1.0, 1.0, 1.0);
    while (*s) {
        // glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *s);
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, *s);
        s++;
    }
}

void showCoordinates(Eigen::Vector3f p, Eigen::Vector3f a, const Eigen::Vector3f offset, int id) {
    // showing coordinates as a fixed entity in the 2D space and
    // not rotating with the mouse
    // glMatrixMode(GL_MODELVIEW);
    // glPushMatrix();
    // gluOrtho2D(0.0, WindowWidth, 0.0, WindowHeight);
    // glLoadIdentity();

    // display object id
    std::string tempID = "ID: " + std::to_string(id);
    const char* cID = tempID.c_str();
    char* c1ID = strdup(cID);
    drawText(c1ID, offset[0], offset[1], offset[2]);

    float x, y, z;
    x = -1000 * p[0] / (2 * gridScale);
    y = 1000 * p[1] / (2 * gridScale);
    z = -1000 * p[2] / (2 * gridScale);

    // display position coordinates
    // convert string to const char*
    // convert const char* to char*
    std::string tempX = "X: " + std::to_string(x);
    const char* c1 = tempX.c_str();
    char* c1x = strdup(c1);
    // drawText(c1x, WindowWidth - 150, WindowHeight - 50, 0);
    drawText(c1x, offset[0], offset[1] - 25, offset[2]);

    std::string tempY = "Y: " + std::to_string(y);
    const char* c2 = tempY.c_str();
    char* c2y = strdup(c2);
    // drawText(c2y, WindowWidth - 150, WindowHeight - 75, 0);
    drawText(c2y, offset[0], offset[1] - 50, offset[2]);

    std::string tempZ = "Z: " + std::to_string(z);
    const char* c3 = tempZ.c_str();
    char* c3z = strdup(c3);
    // drawText(c3z, WindowWidth - 150, WindowHeight - 100, 0);
    drawText(c3z, offset[0], offset[1] - 75, offset[2]);

    // display rotation angle
    a[0] = 180 - a[0];
    a[1] = 180 - a[1];
    a[2] = 180 - a[2];

    std::string tempRx = "Rx: " + std::to_string(a[0]);
    const char* rx = tempRx.c_str();
    char* rx1 = strdup(rx);
    drawText(rx1, offset[0], offset[1] - 100, offset[2]);

    std::string tempRy = "Ry: " + std::to_string(a[1]);
    const char* ry = tempRy.c_str();
    char* ry1 = strdup(ry);
    drawText(ry1, offset[0], offset[1] - 125, offset[2]);

    std::string tempRz = "Rz: " + std::to_string(a[2]);
    const char* rz = tempRz.c_str();
    char* rz1 = strdup(rz);
    drawText(rz1, offset[0], offset[1] - 150, offset[2]);

    // glPopMatrix();
}

// Mouse callback --------------------------------------------------------------

void mouse(int button, int state, int x, int y) {
    MouseX = x;
    MouseY = y;

    switch (button) {
        case WHEEL_UP:
            CameraDistance = (CameraDistance > CAMERA_DISTANCE_MIN ? CameraDistance - 1.0 : CAMERA_DISTANCE_MIN);
            break;
        case WHEEL_DOWN:
            CameraDistance = (CameraDistance < CAMERA_DISTANCE_MAX ? CameraDistance + 1.0 : CAMERA_DISTANCE_MAX);
            break;
    }
    update_camera_location();
}

// Motion callback -------------------------------------------------------------

void motion(int x, int y) {
    CameraLatitude += 180.0 * (double)(x - MouseX) / WindowWidth;
    CameraLongitude += 180.0 * (double)(y - MouseY) / WindowHeight;

    update_camera_location();

    MouseX = x;
    MouseY = y;
}