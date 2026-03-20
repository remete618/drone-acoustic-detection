# Hardware Setup Guide

## Bill of Materials

| Item | Purpose | Approx Cost | Source |
|------|---------|-------------|--------|
| Raspberry Pi 5 (4GB) | Central compute node | ~€80 | rs-online.com |
| 4x MEMS mics (ICS-43434 or SPH0645) on breakout boards | Acoustic array | ~€40 | Pimoroni / Adafruit |
| USB Audio Interface (2ch: Focusrite Scarlett Solo) | Clean 2-channel audio capture | ~€80 | Thomann |
| USB Audio Interface (4ch: Behringer UMC204HD) | Clean 4-channel audio capture | ~€80 | Thomann |
| 2x Cardioid microphones (Behringer C-2 matched pair, or 2x Rode M3 at ~€70 each) | High-fidelity comparison channels | ~€55 | Thomann |
| Tripod + mic stand | Array mounting at fixed height (1.2m) | ~€30 | Amazon |
| Laser distance measurer | Precise drone distance marking | ~€30 | Amazon |
| Measuring tape 50m | Distance markers | ~€15 | Hardware store |
| Pi Camera Module 3 | Visual confirmation / timestamp sync | ~€30 | RS |
| MicroSD 64GB Class 10 | Recording storage | ~€15 | Amazon |
| 3D-printed tetrahedral array mount | 4-mic spatial geometry (5cm spacing) | ~€5 | Printables.com |
| Portable power bank 20,000mAh | Field power | ~€40 | Amazon |
| TI AWR1843BOOST (optional) | 77GHz mmWave radar | ~€300-400 | Mouser / TI |

**Total: ~€400-500** (without radar), **~€700-900** (with radar)

## Microphone Array

### Tetrahedral Configuration

The 4-mic array uses a regular tetrahedron with ~5cm edge length. This provides 3D spatial resolution for direction-of-arrival estimation.

```
        Mic 1 (top)
       /    |    \
      /     |     \
   Mic 2  Mic 3  Mic 4  (base triangle)
```

Mount height: 1.2m on tripod.

### Wiring (MEMS I2S mics to Pi)

For ICS-43434 breakout boards:

| Mic Pin | Pi GPIO |
|---------|---------|
| VDD | 3.3V |
| GND | GND |
| SCK | GPIO 18 (PCM_CLK) |
| WS | GPIO 19 (PCM_FS) |
| SD | GPIO 20 (PCM_DIN) |

For multi-mic I2S, use L/R select pins to multiplex. Or use the USB audio interface approach below.

### USB Audio Interface Approach (Recommended)

Connect MEMS mics via preamp circuits to the USB audio interface inputs. This gives cleaner ADC conversion and works on both Pi and Mac.

**2-channel setup (Focusrite Scarlett Solo):**
- Input 1 (XLR, mic preamp): Cardioid reference mic or MEMS via preamp
- Input 2 (1/4" instrument/line): Requires external preamp for MEMS mic
- Note: Only input 1 has a mic preamp with phantom power

**4-channel setup (Behringer UMC204HD):**
- Input 1-2 (XLR/TRS combo, mic preamps): MEMS mics via preamp or cardioid reference mics
- Input 3-4 (TRS, line-level only): Require external mic preamps for MEMS mics
- Note: Inputs 3-4 have no mic preamp gain or phantom power

## Radar Setup (Optional)

### TI AWR1843BOOST

1. Flash the out-of-box demo firmware using UniFlash
2. Connect CLI port (typically /dev/ttyACM0 on Pi, /dev/cu.usbmodem* on Mac)
3. Connect data port (typically /dev/ttyACM1)
4. The software auto-configures for drone detection profile

### Radar + Acoustic Sync

Both systems use system clock timestamps. For tight sync:
- Start radar and audio capture within the same Python process
- Log timestamps in both data streams
- Post-processing aligns by nearest timestamp

## Field Setup Checklist

- [ ] Array mounted at 1.2m on tripod
- [ ] Distance markers placed at 25m, 50m, 75m, 100m, 150m, 200m
- [ ] Ambient noise measured (SPL meter app)
- [ ] Wind speed noted (Beaufort scale)
- [ ] Temperature noted
- [ ] GPS coordinates logged
- [ ] Test recording made and verified
- [ ] Battery levels checked (Pi power bank, drone batteries)
- [ ] Pi Camera recording visual reference
