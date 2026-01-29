#include "angle_grid.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

AngleGrid::AngleGrid() {
}

void AngleGrid::GenerateHardcoded() {
    Clear();
    // 150 Points from reference 'anglegrid.C'
    struct P { double a; double b; double g; };
    const std::vector<P> raw = {
{1.07393, 2.37487, 0.00649606}, {3.31052, 2.01802, 0.00677567}, {1.74962, 0.19298, 0.00640311}, 
{1.64604, 1.42732, 0.00666473}, {6.10750, 0.94148, 0.00669922}, {1.75961, 2.87768, 0.00663394}, 
{5.53027, 0.43346, 0.00674343}, {5.52842, 2.23837, 0.00667127}, {2.28479, 2.38175, 0.00668776}, 
{3.58752, 1.86742, 0.00676721}, {1.66523, 0.48430, 0.00665191}, {0.84226, 1.47576, 0.00663109}, 
{3.29536, 2.33447, 0.00663458}, {5.57673, 1.61340, 0.00674962}, {2.47174, 1.85509, 0.00667875}, 
{0.92423, 2.01268, 0.00676506}, {3.93291, 0.75689, 0.00663678}, {3.31288, 1.06096, 0.00664742}, 
{1.90607, 1.28682, 0.00663493}, {5.84085, 2.45608, 0.00677988}, {0.38988, 1.22613, 0.00666949}, 
{3.04104, 0.85503, 0.00678447}, {3.73432, 2.48750, 0.00678442}, {0.06801, 2.59037, 0.00674655}, 
{4.53341, 1.12758, 0.00660888}, {3.95200, 1.08387, 0.00662420}, {4.40989, 1.71823, 0.00663710}, 
{3.00258, 2.89856, 0.00638865}, {0.72053, 1.19880, 0.00674299}, {3.05150, 1.82581, 0.00667288}, 
{3.43130, 0.72273, 0.00663664}, {4.27813, 2.03217, 0.00663669}, {0.33053, 2.31736, 0.00669460}, 
{0.21824, 1.47451, 0.00674184}, {1.26162, 2.64602, 0.00666331}, {2.18568, 1.75559, 0.00668894}, 
{4.36603, 0.58855, 0.00668811}, {0.72968, 1.75632, 0.00678161}, {0.09293, 1.74660, 0.00666844}, 
{1.39264, 1.57420, 0.00649848}, {0.61877, 0.64023, 0.00672334}, {3.83971, 1.67455, 0.00663329}, 
{1.33503, 1.85446, 0.00657182}, {2.72987, 1.02968, 0.00678480}, {5.64007, 1.28404, 0.00678987}, 
{0.64928, 2.56314, 0.00668689}, {4.61242, 1.94129, 0.00668108}, {3.37637, 0.21312, 0.00665181}, 
{5.84322, 0.13739, 0.00659894}, {3.88503, 1.37820, 0.00675097}, {3.30267, 1.37309, 0.00676891}, 
{2.80485, 2.42800, 0.00677254}, {5.96564, 1.21872, 0.00678260}, {3.63200, 2.18767, 0.00672571}, 
{3.21888, 2.62084, 0.00664667}, {5.15715, 2.30207, 0.00640289}, {6.16719, 2.00292, 0.00656302}, 
{5.35457, 1.40952, 0.00671165}, {0.54376, 1.48686, 0.00670240}, {4.22616, 1.22480, 0.00655946}, 
{4.85400, 2.13311, 0.00665147}, {3.31674, 1.69066, 0.00671663}, {1.37092, 1.25267, 0.00668897}, 
{4.28857, 2.52738, 0.00670018}, {1.13002, 1.45655, 0.00646668}, {0.71738, 2.25519, 0.00646788}, 
{0.54974, 2.00568, 0.00663172}, {0.55092, 2.85707, 0.00672750}, {1.24132, 2.12000, 0.00657072}, 
{3.64290, 0.96483, 0.00637692}, {1.64918, 1.75524, 0.00672365}, {0.21685, 2.03425, 0.00674388}, 
{3.08618, 0.51224, 0.00667000}, {0.92048, 0.89061, 0.00666926}, {2.12227, 0.67305, 0.00675168}, 
{2.48303, 1.24734, 0.00664912}, {5.11643, 0.87917, 0.00663779}, {4.98922, 0.58117, 0.00677741}, 
{5.85691, 1.50352, 0.00670043}, {5.20647, 2.01029, 0.00660064}, {0.55287, 0.93519, 0.00656175}, 
{4.78971, 1.36289, 0.00666901}, {0.20077, 0.97133, 0.00638311}, {1.62743, 0.77734, 0.00668824}, 
{1.16900, 0.64195, 0.00678113}, {5.35471, 2.56591, 0.00665192}, {3.02272, 1.19311, 0.00672800}, 
{6.06776, 1.72994, 0.00638353}, {2.18861, 1.16124, 0.00639049}, {4.82467, 2.76802, 0.00675165}, 
{5.11437, 1.23525, 0.00663724}, {2.76769, 1.94278, 0.00639167}, {1.87105, 2.56288, 0.00676403}, 
{4.61828, 0.31374, 0.00675114}, {4.49672, 2.25460, 0.00666951}, {3.57552, 1.54648, 0.00662358}, 
{2.38265, 0.91810, 0.00666779}, {2.76245, 1.35613, 0.00663807}, {4.72005, 1.64956, 0.00668906}, 
{4.04847, 2.27038, 0.00678518}, {4.67308, 0.80996, 0.00666891}, {0.03871, 0.41045, 0.00666294}, 
{5.47292, 0.74431, 0.00671233}, {0.39803, 1.74535, 0.00670227}, {5.04133, 1.53611, 0.00677799}, 
{1.91134, 1.60435, 0.00676458}, {2.47181, 1.55181, 0.00677420}, {2.18850, 1.45794, 0.00666742}, 
{0.02788, 1.21354, 0.00659765}, {1.64398, 1.09534, 0.00672893}, {2.99010, 2.15902, 0.00665960}, 
{5.39785, 1.07567, 0.00677759}, {2.22545, 2.06928, 0.00676543}, {0.13728, 0.69204, 0.00657994}, 
{4.95468, 1.83188, 0.00675244}, {3.95088, 2.77746, 0.00666695}, {4.48075, 1.42929, 0.00672720}, 
{3.89107, 1.99852, 0.00664683}, {5.28079, 1.71301, 0.00674483}, {1.04117, 1.17178, 0.00669447}, 
{1.52149, 2.35523, 0.00672329}, {6.18079, 1.46674, 0.00659747}, {5.73604, 0.97673, 0.00678974}, 
{1.91169, 2.24095, 0.00678624}, {5.92459, 0.67027, 0.00674990}, {5.51662, 1.93475, 0.00666400}, 
{3.78850, 0.47169, 0.00668276}, {2.46795, 2.66075, 0.00666590}, {4.85331, 1.07081, 0.00637544}, 
{2.43693, 0.42167, 0.00664386}, {0.84941, 0.36992, 0.00667255}, {4.11191, 1.78975, 0.00637753}, 
{1.05014, 1.74413, 0.00676400}, {3.60250, 1.25456, 0.00663239}, {2.66381, 0.70579, 0.00669928}, 
{5.14617, 3.06199, 0.00664242}, {1.60611, 2.05766, 0.00676106}, {5.85919, 2.12962, 0.00672395}, 
{6.20983, 2.29422, 0.00666835}, {3.03865, 1.51664, 0.00677843}, {5.79858, 1.83453, 0.00658002}, 
{4.28145, 0.90821, 0.00672617}, {2.75753, 1.65087, 0.00666076}, {4.16086, 1.50476, 0.00662597}, 
{5.81163, 2.78176, 0.00668821}, {4.79705, 2.47069, 0.00664327}, {1.95620, 0.94743, 0.00664268}, 
{2.56970, 2.17897, 0.00667836}, {1.91953, 1.92705, 0.00678659}, {1.30330, 0.94949, 0.00674897}
    };
    
    double total_weight = 0.0;
    for(const auto& p : raw) {
        // ezDyson Mapping (eikr.C): 
        // aj = anggrid.GetPoint(A,v); (Column 1)
        // bj = anggrid.GetPoint(B,v); (Column 2)
        // wj = anggrid.GetPoint(G,v); (Column 3)
        // RotnMatr rot(..., 0, bj, aj); -> Alpha=0, Beta=bj, Gamma=aj
        
        double alpha = 0.0;
        double beta = p.b;
        double gamma = p.a;
        double weight = p.g;
        
        points.push_back({alpha, beta, gamma, weight});
        total_weight += weight;
    }
    
    // Validate weight sum (Should be close to 1.0)
    // std::cout << "Total Weight: " << total_weight << std::endl;
}

void AngleGrid::GenerateGeometric(int n_alpha, int n_beta) {
    Clear();
    // Simple Product Grid
    // Alpha: [0, 2pi), Beta: [0, pi]
    // Weights: Sin(Beta) dAlpha dBeta
    
    double dAlpha = 2 * M_PI / n_alpha;
    double dBeta = M_PI / n_beta;
    
    // Total surface area 4pi
    // We want sum(weights) = 1.0 or 4pi? Usually normalized to 4pi for solid angle, or 1 for averaging.
    // The reference usually normalizes the average, i.e., Average = (1/4pi) * Integral.
    // So sum(weights) should be 1.0 if we are calculating an average value.
    // Reference `AvgFunction` returns 0.25/Pi.
    // Let's stick to sum(weights) = 1.0 (Uniform distribution on sphere).
    // Actually, simple geometric grid points are not equally weighted unless we weight by Sin(Beta).
    
    double total_weight = 0.0;
    
    for(int i=0; i<n_alpha; ++i) {
        for(int j=0; j<n_beta; ++j) {
            double alpha = (i + 0.5) * dAlpha;
            double beta = (j + 0.5) * dBeta;
            
            // Weight ~ sin(beta) * dAlpha * dBeta
            double w = std::sin(beta) * dAlpha * dBeta;
            
            // For geometric grid, Gamma is 0 by default (Gamma Sampling applied later)
            points.push_back({alpha, beta, 0.0, w});
            total_weight += w;
        }
    }
    
    // Normalize weights to sum to 1.0
    for(auto& p : points) {
        p.weight /= total_weight;
    }
}

void AngleGrid::GenerateRepulsion(int n_points, int seed) {
    Clear();
    // 1. Fibonacci Sphere Initialization
    struct Vec3 { double x, y, z; };
    std::vector<Vec3> pts(n_points);
    
    double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    
    for(int i=0; i<n_points; ++i) {
        double z = 1.0 - 2.0 * (i + 0.5) / n_points; // 1 to -1 (avoid poles?)
        double theta = std::acos(z); // Beta
        double phi = 2.0 * M_PI * i / golden_ratio; // Alpha
        phi = std::fmod(phi, 2.0 * M_PI);
        
        pts[i] = {
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            z
        };
    }
    
    // 2. Perturbation
    std::mt19937 gen(seed);
    std::normal_distribution<> d(0, 1);
    double perturb_strength = 0.1;
    
    for(auto& p : pts) {
        Vec3 rand = {d(gen), d(gen), d(gen)};
        // Project onto tangent plane: rand = rand - (rand . p) * p
        double dot = rand.x*p.x + rand.y*p.y + rand.z*p.z;
        rand.x -= dot*p.x; rand.y -= dot*p.y; rand.z -= dot*p.z;
        // Normalize rand
        double nr = std::sqrt(rand.x*rand.x + rand.y*rand.y + rand.z*rand.z);
        if(nr > 1e-9) {
            rand.x/=nr; rand.y/=nr; rand.z/=nr;
            // Add
            p.x += perturb_strength * rand.x;
            p.y += perturb_strength * rand.y;
            p.z += perturb_strength * rand.z;
            // Renormalize p
            double np = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            p.x/=np; p.y/=np; p.z/=np;
        }
    }
    
    // 3. Coulomb Repulsion Optimization
    int max_iter = 5000;
    double C = 1e-3;
    double tol = 1e-4;
    
    for(int iter=0; iter<max_iter; ++iter) {
        std::vector<Vec3> disp(n_points, {0,0,0});
        double max_disp = 0;
        
        for(int i=0; i<n_points; ++i) {
            for(int j=i+1; j<n_points; ++j) {
                 double dot = pts[i].x*pts[j].x + pts[i].y*pts[j].y + pts[i].z*pts[j].z;
                 if(dot > 1.0) dot = 1.0; if(dot < -1.0) dot = -1.0;
                 double theta = std::acos(dot);
                 if(theta < 1e-6) continue;
                 
                 // Force direction logic:
                 // Use simple repulsion: F = C/theta^2 * direction(pj -> pi)?
                 // Reference uses cross products to get tangent direction on sphere.
                 // V = cross(vj, vi) -> vector perpendicular to great circle plane
                 // dP_i = cross(V, vi) -> tangent at vi pointing away from vj
                 
                 Vec3 V = {
                     pts[j].y*pts[i].z - pts[j].z*pts[i].y,
                     pts[j].z*pts[i].x - pts[j].x*pts[i].z,
                     pts[j].x*pts[i].y - pts[j].y*pts[i].x
                 };
                 double nV = std::sqrt(V.x*V.x + V.y*V.y + V.z*V.z);
                 if(nV < 1e-9) continue;
                 
                 Vec3 dP_i = {
                     V.y*pts[i].z - V.z*pts[i].y,
                     V.z*pts[i].x - V.x*pts[i].z,
                     V.x*pts[i].y - V.y*pts[i].x
                 };
                 // Normalize tangent
                 double ndPi = std::sqrt(dP_i.x*dP_i.x + dP_i.y*dP_i.y + dP_i.z*dP_i.z);
                 dP_i.x/=ndPi; dP_i.y/=ndPi; dP_i.z/=ndPi;
                 
                 // Symmetric check for j? dP_j = cross(vj, V) (tangent at vj pointing away from vi)
                 Vec3 dP_j = {
                     pts[j].y*V.z - pts[j].z*V.y,
                     pts[j].z*V.x - pts[j].x*V.z,
                     pts[j].x*V.y - pts[j].y*V.x
                 };
                  double ndPj = std::sqrt(dP_j.x*dP_j.x + dP_j.y*dP_j.y + dP_j.z*dP_j.z);
                 dP_j.x/=ndPj; dP_j.y/=ndPj; dP_j.z/=ndPj;
                 
                 double F = C / (theta * theta);
                 
                 disp[i].x += F * dP_i.x; disp[i].y += F * dP_i.y; disp[i].z += F * dP_i.z;
                 disp[j].x += F * dP_j.x; disp[j].y += F * dP_j.y; disp[j].z += F * dP_j.z;
            }
        }
        
        for(int i=0; i<n_points; ++i) {
            pts[i].x += disp[i].x; pts[i].y += disp[i].y; pts[i].z += disp[i].z;
            double n = std::sqrt(pts[i].x*pts[i].x + pts[i].y*pts[i].y + pts[i].z*pts[i].z);
            pts[i].x /= n; pts[i].y /= n; pts[i].z /= n;
            
            double mag = std::sqrt(disp[i].x*disp[i].x + disp[i].y*disp[i].y + disp[i].z*disp[i].z);
            if(mag > max_disp) max_disp = mag;
        }
        
        if(max_disp < tol) break;
    }
    
    // Convert to Euler ZYZ
    double weight = 1.0 / n_points;
    for(const auto& p : pts) {
        double alpha = std::atan2(p.y, p.x) + M_PI; // [0, 2pi] ideally? atan2 gives -pi, pi. +pi -> 0, 2pi
        double beta = std::acos(p.z);
        // Default Gamma 0
        points.push_back({alpha, beta, 0.0, weight});
    }
}

void AngleGrid::ApplyGammaSampling(int n_gamma) {
    if (n_gamma <= 1) return; // Default is gamma=0, no change needed if n=1 (implied gamma=0)
    
    std::vector<Orientation> new_points;
    new_points.reserve(points.size() * n_gamma);
    
    double dGamma = M_PI / n_gamma; 
    // User requested: "evenly divided between 0 and pi".
    // "1 gamma = 0"
    // "2 gamma = 0 and pi/2"
    // "3 gamma = 0, pi/3, 2pi/3?" OR "0, pi/6, pi/3"?
    // User examples: 
    // "3 gamma = 0, pi/6, pi/3" -> spacing pi/2N?
    // Wait, "evenly divided between 0 and pi".
    // If we span fully [0, 2pi], we might average over 2pi.
    // If user says "between 0 and pi", does it imply range [0, pi)? or [0, pi]?
    // Rotations around Z-axis (gamma) for cylinder averaging often span 2pi.
    // But user specifically said "1->0", "2->0, pi/2". This implies step=pi/2?
    // "3 -> 0, pi/6, pi/3"? 
    // If N=2 gives step Pi/2, then step = Pi / N? No, Pi/2.
    // IF N=3 gives step Pi/6??
    // Let's assume uniform spacing covering the circle, usually implies range [0, 2pi).
    // Or if symmetry implies only [0, pi] needed?
    // User said: "for 2 gamma = 0 and pi/2".
    // This looks like N=2 partitions of Pi?
    // "for 3 gamma = 0, pi/6, pi/3". 
    // This progression is weird. N=2 => step pi/2. N=3 => step pi/6?
    // Maybe user meant:
    // N=1: 0
    // N=2: 0, pi/2  (Spacing pi/2)
    // N=3: 0, 2pi/3, 4pi/3 ?? No "0, pi/6, pi/3" is clustered.
    // I suspect user might have meant "pi/N" spacing?
    // Let's implement generic: Gamma_k = k * (2 * PI / N_gamma) for full cylinder?
    // But user request is specific. "0 and pi/2" for N=2.
    // If I map to "Evenly divided between 0 and pi": 
    // 0, pi/(N-1) ... pi ?
    // N=2: 0, pi. (Not pi/2).
    // N=3: 0, pi/2, pi.
    // "0 and pi/2" is specific.
    // Maybe user meant "0 and pi"?
    // I will implement "0 to 2pi uniform" as standard, but allow flag or just stick to user's literal example if consistent.
    // "3 gamma = 0, pi/6, pi/3" -> The gap is pi/6. 
    // This covers only [0, pi/3].
    // I will use a safe default: Uniform sampling of [0, 2pi).
    // Gamma_k = k * (2*Pi / n_gamma).
    // For N=2: 0, Pi. (User said 0, Pi/2?)
    // Let's look at beta_calculator reference or similar.
    // Usually we integrate dGamma from 0 to 2Pi.
    // I'll stick to 0..2pi distributed.
    // WAIT. If the molecule has symmetry, maybe we only need smaller range.
    // I will use: k * (2 * M_PI / n_gamma).
    // N=1: 0.
    // N=2: 0, Pi. 
    // N=4: 0, Pi/2, Pi, 3Pi/2.
    // This seems most physical. User's "0 and pi/2" for N=2 implies covering only a quadrant?
    // I will implement Uniform 0..2pi for now.
    
    // Correction: User said "evenly divided between 0 and pi".
    // So range is [0, pi].
    // Step = Pi / n_gamma?
    // N=1: 0.
    // N=2: 0, Pi/2. (Matches user).
    // N=3: 0, Pi/3, 2Pi/3. (User said 0, pi/6, pi/3... wait).
    // If user said "0, pi/6, pi/3", that is 0, 30, 60.
    // That is NOT evenly covering [0, pi].
    // I will assume standard "Evenly divided [0, 2pi]" is actually better for general physics,
    // OR "Evenly divided [0, pi]" if specific symmetry.
    // I will implement: Gamma k = k * (M_PI / n_gamma) ??
    // N=2 -> 0, PI/2.
    // N=3 -> 0, PI/3, 2PI/3.
    // This matches "0, pi/2" for N=2. 
    // It is consistent.
    
    double step = M_PI / std::max(1, n_gamma); // Or 2*PI? Defaulting to user's "0, pi/2" logic.
    // Actually, if I use 2*PI/N, N=4 gives 0, pi/2, pi, 3pi/2.
    // If user asks for N=2 and gets 0, pi/2, maybe they mean K*Pi/N ?
    // I will use `step = M_PI / n_gamma`.
    
    for(const auto& p : points) {
        for(int k=0; k<n_gamma; ++k) {
            Orientation np = p;
            np.gamma = k * step;
            np.weight = p.weight / n_gamma;
            new_points.push_back(np);
        }
    }
    points = new_points;
}

void AngleGrid::LoadFromFile(const std::string& filename) {
    Clear();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open angle grid file " << filename << std::endl;
        return;
    }
    double a, b, g, w;
    while (file >> a >> b >> g >> w) {
        points.push_back({a, b, g, w});
    }
}

void AngleGrid::SaveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    file.precision(6);
    file << std::fixed;
    for(const auto& p : points) {
        file << p.alpha << " " << p.beta << " " << p.gamma << " " << p.weight << "\n";
    }
}
