#include <array>
#include <iostream>

#include "attacks.h"
#include "types.h"

namespace Chess {

namespace Attacks {

std::array<Bitboard, Square::COUNT> initJumperAttacks(Bitboard (*slowAttacks)(Square square));
std::array<Bitboard, Square::COUNT> initSliderMasks(bool bishop);
std::array<int     , Square::COUNT> initSliderShifts(std::array<Bitboard, Square::COUNT> &masks);

std::array<std::array<Bitboard, Square::COUNT>, Square::COUNT> initMaskBB(bool extended);

Bitboard knightAttacksSlow(Square square);
Bitboard sliderAttacksSlow(Square square, Bitboard blocker, bool bishop);
Bitboard kingAttacksSlow(Square square);

std::array<Bitboard, Square::COUNT> knightAttacks;
std::array<Bitboard, Square::COUNT> kingAttacks;

std::array<Bitboard, Square::COUNT> bishopMasks;
std::array<Bitboard, Square::COUNT> rookMasks;

std::array<int, Square::COUNT> bishopShifts;
std::array<int, Square::COUNT> rookShifts;

std::array<std::array<Bitboard,  512>, Square::COUNT> bishopTable;
std::array<std::array<Bitboard, 4096>, Square::COUNT> rookTable;

std::array<std::array<Bitboard, Square::COUNT>, Square::COUNT> betweenBB;
std::array<std::array<Bitboard, Square::COUNT>, Square::COUNT> lineBB;

const std::array<uint64_t, Square::COUNT> bishopMagics {
    5226430236606988800ULL, 5226430236606988800ULL, 308593332580267072ULL, 362825647322173444ULL,
    4804041364242448ULL, 1297319404912069120ULL, 13840126872677277704ULL, 9385783651125955076ULL,
    497546326180096ULL, 9269569220503421568ULL, 603486765363380480ULL, 578716967385432082ULL,
    2341893951620917249ULL, 18050751216550088ULL, 4508019168678144ULL, 5188182646610464832ULL,
    9241405540151896320ULL, 9227981601999814788ULL, 1227512407929136514ULL, 4612811923233210374ULL,
    9305563007090163714ULL, 74318469126033409ULL, 2814756243185666ULL, 4638753796905504768ULL,
    7134938761412478984ULL, 298367890810152968ULL, 1333633938320786432ULL, 580968749981503520ULL,
    759146826082336768ULL, 9335962577899389952ULL, 1153207382468461568ULL, 2594364755991068928ULL,
    149489635276361856ULL, 444176738839572488ULL, 92367582660608ULL, 144117389247119489ULL,
    594475726349598980ULL, 9818005519490941072ULL, 9806614613989592200ULL, 2253999644442756ULL,
    73469779840599044ULL, 291440110338634ULL, 666852781251891200ULL, 1306057386733148161ULL,
    2542083780903936ULL, 72726102920941824ULL, 4638712171014922753ULL, 13523997318722625ULL,
    10664880228777133120ULL, 282579052932096ULL, 9800959790733590561ULL, 9943986186370548480ULL,
    87821344003325960ULL, 4611721220790976512ULL, 38423567738863618ULL, 9033727145443328ULL,
    9297226237383032836ULL, 288793609607651330ULL, 180319954210426880ULL, 72092784858793993ULL,
    9800009950394188296ULL, 5764607696007857664ULL, 289928030830330890ULL, 4505807442154384ULL,
};


const std::array<uint64_t, Square::COUNT> rookMagics {
    2341873049361449088ULL, 18015016991064128ULL, 648536350844797185ULL, 1224996695127885828ULL,
    9943983307635560460ULL, 4755816599868418053ULL, 288247985535582360ULL, 4755810158298006017ULL,
    1193594782623776768ULL, 141012374872064ULL, 9233083473575583744ULL, 315392951997960192ULL,
    2392554490300418ULL, 563293888577656ULL, 600104685353119748ULL, 140824738267392ULL,
    18014948274749472ULL, 22518273551646720ULL, 9516110960973087376ULL, 4611827305805778944ULL,
    1513350762241984512ULL, 563499751187456ULL, 2484896564119568ULL, 1154401447266287740ULL,
    9962842536949071875ULL, 4613744305268891648ULL, 2306265505148829712ULL, 162692648208973984ULL,
    18652117401600128ULL, 865254361877714960ULL, 74590873123422220ULL, 72066690778694657ULL,
    9259471271882653824ULL, 2387048677442076673ULL, 9799867974654366466ULL, 2305852909666963456ULL,
    4644354303985664ULL, 1154482813274161664ULL, 289919261479013409ULL, 5188149261845405825ULL,
    72198606410514432ULL, 22588371184451632ULL, 71468792741910ULL, 4639041872031973384ULL,
    9241949403303510048ULL, 18225642452025604ULL, 11547792428960448513ULL, 883832528535420940ULL,
    1020621676871808ULL, 4755836396244076672ULL, 144256063020991104ULL, 5207321538986112ULL,
    13844079548482191616ULL, 39584566214784ULL, 153140013910541312ULL, 4911179797079624192ULL,
    288559131210551810ULL, 288559131210551810ULL, 55113020883201ULL, 2594108638458347529ULL,
    10376856663482441802ULL, 1193735393409894405ULL, 2260664794027652ULL, 18225796817887362ULL,
};

Bitboard getRelevantEdges(Square square) {
    Bitboard bb            = Bitboard(square);
    Bitboard relevantEdges = Bitboard(0);
    std::array<Bitboard, 4> edges = {Bitboard(Rank(Rank::RANK_1)), Bitboard(Rank(Rank::RANK_8)), 
                                     Bitboard(File(File::A_FILE)), Bitboard(File(File::H_FILE))};

    for (size_t i = 0; i < edges.size(); i++)
        if (!(edges[i] & bb))
            relevantEdges |= edges[i];

    return relevantEdges;
}

template<int ATTACK_TABLE_SIZE>
std::array<std::array<Bitboard, ATTACK_TABLE_SIZE>, Square::COUNT> initAttackTables() {
    static_assert(ATTACK_TABLE_SIZE == 4096 || ATTACK_TABLE_SIZE == 512, "Unexpected attacktable size");

    constexpr bool BISHOP = ATTACK_TABLE_SIZE == 512;

    std::array<std::array<Bitboard, ATTACK_TABLE_SIZE>, Square::COUNT> attackTable;

    const std::array<Bitboard, Square::COUNT> &masks  = BISHOP ? bishopMasks  : rookMasks;
    const std::array<uint64_t, Square::COUNT> &magics = BISHOP ? bishopMagics : rookMagics;
    const std::array<int     , Square::COUNT> &shifts = BISHOP ? bishopShifts : rookShifts;

    
    for (Square square = Square::H1; square < Square::COUNT; ++square) {
        Bitboard blockers = Bitboard(0);

        Bitboard mask  = masks [square];
        uint64_t magic = magics[square];
        int      shift = shifts[square];

        do {
            int index = ((mask & blockers) * magic) >> shift;

            attackTable[square][index] = sliderAttacksSlow(square, blockers, BISHOP);

            blockers = (blockers - mask) & mask;
        } while (blockers);
    }

    return attackTable;
}

void init() {
    knightAttacks = initJumperAttacks(&knightAttacksSlow);
    kingAttacks   = initJumperAttacks(&kingAttacksSlow);

    bishopMasks = initSliderMasks(true);
    rookMasks   = initSliderMasks(false);

    bishopShifts = initSliderShifts(bishopMasks);
    rookShifts   = initSliderShifts(rookMasks);

    bishopTable = initAttackTables< 512>();
    rookTable   = initAttackTables<4096>();

       lineBB = initMaskBB(true );
    betweenBB = initMaskBB(false);
}

std::array<Bitboard, Square::NONE> initJumperAttacks(Bitboard (*slowAttacks)(Square square)) {
    std::array<Bitboard, Square::NONE> out = {};

    for (Square square = Square::H1; square < Square::NONE; ++square) {
        out[int(square)] = slowAttacks(square);
    }

    return out;
}

Bitboard knightAttacksSlow(Square square) {
    Bitboard bb      = Bitboard(square);
    Bitboard attacks = Bitboard(0);

    if (square.getFile() > File::H_FILE)
        attacks |= bb << 15 | bb >> 17;

    if (square.getFile() > File::G_FILE)
        attacks |= bb << 6 | bb >>  10;

     if (square.getFile() < File::A_FILE)
        attacks |= bb << 17 | bb >> 15;

    if (square.getFile() < File::B_FILE)
        attacks |= bb << 10 | bb >>  6;

    return attacks;
}

Bitboard kingAttacksSlow(Square square) {
    Bitboard bb      = Bitboard(square);
    Bitboard attacks = bb << 8 | bb >> 8;

    if (square.getFile() > File::H_FILE)
        attacks |= bb << 7 | bb >> 9 | bb >> 1;

    if (square.getFile() < File::A_FILE)
        attacks |= bb << 9 | bb >> 7 | bb << 1;


    return attacks;
}

Bitboard sliderAttacksSlow(Square square, Bitboard blocker, bool bishop) {
    Bitboard bb           = Bitboard(square);
    Bitboard attacks      = Bitboard(0);
    Bitboard releventEdge = getRelevantEdges(square);

    const int  firstDirection = bishop ? 9 : 1;
    const int secondDirection = bishop ? 7 : 8;

    std::array<int, 4> shifts = {firstDirection, secondDirection, -firstDirection, -secondDirection};

    auto shift = [](Bitboard bb, int amount) { return amount >= 0 ? bb << amount : bb >> -amount; };

    for (size_t i = 0; i < shifts.size(); i++) {
        Bitboard directionAttacks = shift(bb, shifts[i]);

        if (!(kingAttacksSlow(square) & directionAttacks))
            continue;

        while (!(directionAttacks & (releventEdge | blocker)))
            directionAttacks |= shift(directionAttacks, shifts[i]);

        attacks |= directionAttacks;
    }

    return attacks;
}

std::array<Bitboard, Square::COUNT> initSliderMasks(bool bishop) {
    Bitboard edges = Bitboard(Rank(Rank::RANK_1)) | Bitboard(Rank(Rank::RANK_8)) | Bitboard(File(File::A_FILE)) | Bitboard(File(File::H_FILE));
    std::array<Bitboard, Square::COUNT> masks;

    for (Square square = Square::H1; square < Square::COUNT; ++square) {
        masks[square] = sliderAttacksSlow(square, Bitboard(0), bishop) & ~getRelevantEdges(square);
    }

    return masks;
}

std::array<int, Square::COUNT> initSliderShifts(std::array<Bitboard, Square::COUNT> &masks) {
    std::array<int, Square::COUNT> shifts;

    for (Square square = Square::H1; square < Square::COUNT; ++square) {
        shifts[square] = 64 - __builtin_popcountll(masks[square]);
    }

    return shifts;
}

std::array<std::array<Bitboard, Square::COUNT>, Square::COUNT> initMaskBB(bool extended) {
    std::array<std::array<Bitboard, Square::COUNT>, Square::COUNT> masksBB;

    for (Square sq1 = Square::H1; sq1 < Square::NONE; ++sq1) {
        for (Square sq2 = Square::H1; sq2 < Square::NONE; ++sq2) {
            Bitboard occupied = extended ? 0 : Bitboard(sq1) | Bitboard(sq2);
            Bitboard mask     = 0;

            if (Chess::getBishopAttacks(sq1, occupied) & Bitboard(sq2))
                mask = Chess::getBishopAttacks(sq1, occupied) & Chess::getBishopAttacks(sq2, occupied);
            else if (Chess::getRookAttacks(sq1, occupied) & Bitboard(sq2))
                mask = Chess::getRookAttacks(sq1, occupied) & Chess::getRookAttacks(sq2, occupied);

            masksBB[sq1][sq2] = mask;
        }
    }

    return masksBB;
}

} // Namespace Attacks
 
} // Namespace Chess
